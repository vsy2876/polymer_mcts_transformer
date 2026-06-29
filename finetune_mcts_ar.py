"""MCTS-based finetuning for PSELFIES autoregressive model."""
import os
import sys
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import tempfile

# Force all Python temporary sandboxes to use PACE Phoenix Scratch space
if "SCRM" in os.environ:
    tempfile.tempdir = os.environ["SCRM"]
elif "SCRATCH" in os.environ:
    tempfile.tempdir = os.environ["SCRATCH"]

import csv
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

from tokenizer_pselfies import PSELFIESTokenizer
from model_ar import PSELFIESLanguageModel
from collections import deque
import selfies
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import traceback
sys.path.append('/storage/home/hcoda1/2/vyadav68/.conda/envs/polymers/share/RDKit/Contrib/SA_Score')
import sascorer

# ============================================================================
# CONFIGURATION
# ============================================================================
SIGMA_START = 0.5
SIGMA_END = 0.15
SIGMA_ANNEAL_START = 2000
SIGMA_ANNEAL_END = 4000
RESUME_FROM_ITER = None  # Set to an integer step to resume from scratch, or None to start fresh

# ============================================================================


@dataclass
class Node:
    """MCTS tree node."""
    token_id: int
    parent: Optional["Node"]
    children: dict
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 0.0


class MCTSSearch:
    """AlphaZero-style MCTS for sequence generation."""

    def __init__(
        self,
        policy_model: PSELFIESLanguageModel,
        reference_model: PSELFIESLanguageModel,
        tokenizer: PSELFIESTokenizer,
        target_min: float,
        target_max: float,
        n_simulations: int = 50,
        c_puct: float = 1.25,
        top_k: int = 10,
        tau: float = 1.0,
        max_length: int = 128,
        save_dir: str = "checkpoints/mcts",
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.tokenizer = tokenizer
        self.target_min = target_min
        self.target_max = target_max
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.tau = tau
        self.max_length = max_length
        self.device = next(policy_model.parameters()).device
        self.log_path = os.path.join(save_dir, "sequence_log.csv")
        self.true_oracle_calls = 0 # Track true oracle calls for evaluation purposes

        # Create the folder and initialize the CSV headers exactly once at launch.
        # This removes slow metadata checks from your hot evaluation/MCTS loops.
        if RESUME_FROM_ITER is None or not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])

    def puct_score(self, parent: Node, child: Node) -> float:
        """Compute PUCT score for child selection."""
        u = self.c_puct * child.P * math.sqrt(parent.N) / (1 + child.N)
        return child.Q + u

    def select(self, root: Node) -> Node:
        """Select leaf node using PUCT."""
        node = root
        while node.children:
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = self.puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    def expand(self, node: Node, sequence: list, target_raw: float) -> tuple:
        """Expand leaf node using policy network with compile-friendly static shapes."""
        # Pad the sequence to max_length to maintain a static tensor shape for torch.compile
        padded_sequence = sequence + [self.tokenizer.pad_id] * (self.max_length - len(sequence))
        seq_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=self.device)
        
        # Accept target_raw directly to preserve distribution parity across both execution states
        target_tensor = torch.tensor([target_raw], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, values = self.policy(seq_tensor, property=target_tensor)

            # Isolate predictions at the actual active sequence terminal index, ignoring trailing padding
            active_idx = len(sequence) - 1
            predicted_value = values[0, active_idx].item()
            logits = logits[0, active_idx, :]  # (vocab_size,)
            
            # Prohibit selection of administrative and utility tokens
            logits[self.tokenizer.pad_id] = -float('inf')
            logits[self.tokenizer.mask_id] = -float('inf')
            logits[self.tokenizer.unk_id] = -float('inf')
            logits[self.tokenizer.bos_id]  = -float('inf')
            
            probs = F.softmax(logits, dim=-1)

        # Top-k expansion
        topk_probs, topk_ids = torch.topk(probs, self.top_k)
        topk_probs = topk_probs / topk_probs.sum()  # Renormalize

        # DIRICHLET NOISE INJECTION
        if node.parent is None:
            epsilon = 0.25 
            alpha = torch.full((self.top_k,), 0.3, device=self.device)
            noise = torch.distributions.Dirichlet(alpha).sample()
            topk_probs = (1 - epsilon) * topk_probs + epsilon * noise
            topk_probs = topk_probs / topk_probs.sum()

        for token_id, prob in zip(topk_ids.tolist(), topk_probs.tolist()):
            child = Node(
                token_id=token_id,
                parent=node,
                children={},
                P=prob,
            )
            node.children[token_id] = child

        return node, predicted_value

    def pselfies_to_psmiles(self, pselfies: str) -> str:
        """
        Convert PSELFIES to raw p-SMILES strings preserving wildcard handles 
        for graph-level downstream polymerization reactions.
        """
        try:
            # 1. Strip structural sequence padding tokens cleanly
            pselfies_clean = pselfies.replace("<BOS>", "").replace("<EOS>", "").replace("<PAD>", "").strip()
            
            # 2. Pure grammar decoding (leaves [At] tags intact as reactive graph handles)
            smiles = selfies.decoder(pselfies_clean)
            return smiles if smiles else ""
        except Exception:
            return ""

    def _execute_composite_calculation(self, tmpdir, smiles_string, prefix):
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None: return None
            
            mol = Chem.AddHs(mol)
            
            # 1. Very Fast 3D Embedding Check
            if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=10) != 0:
                return None
                
            # 2. Fast Force Field Pre-Optimization
            try:
                # UFF is often faster and more robust for initial cleanup than MMFF
                if AllChem.UFFOptimizeMolecule(mol, maxIters=200) != 0:
                    return None  # Didn't converge quickly
            except Exception:
                return None
                
            xyz_path = os.path.join(tmpdir, f"{prefix}_input.xyz")
            Chem.MolToXYZFile(mol, xyz_path)

            # 3. GFN2-xTB Optimization (Aggressive Timeouts & Loose Criteria)
            # Use --opt loose and limit cycles to prevent hanging on floppy molecules
            opt_cmd = ["xtb", f"{prefix}_input.xyz", "--gfn", "2", "--opt", "loose", "--cycles", "100", "--parallel", "12"]
            
            try:
                # Short timeout. If it takes longer than 45s, it's likely a bad structure anyway.
                result_opt = subprocess.run(opt_cmd, cwd=tmpdir, capture_output=True, text=True, check=False, timeout=45)
            except subprocess.TimeoutExpired:
                return None
                
            if result_opt.returncode != 0:
                return None

            opt_xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
            if not os.path.exists(opt_xyz_path):
                return None

            # 4. GFN2-xTB Single Point for Gap (Very Fast)
            try:
                result_gxtb = subprocess.run(
                    ["xtb", "xtbopt.xyz", "--gxtb", "--parallel", "12"], # Run on the optimized geometry
                    cwd=tmpdir, capture_output=True, text=True, check=False, timeout=30
                )
            except subprocess.TimeoutExpired:
                return None

            if result_gxtb.returncode != 0:
                return None

            output_text = result_gxtb.stdout
            match = re.search(r"(?:HOMO-LUMO\s+gap|HL-Gap)\s*[:\/\s\w]*\s*([-+]?\d*\.\d+|\d+)", output_text, re.IGNORECASE)
            if match:
                return float(match.group(1))
                
            # Fallback parsing
            for line in output_text.splitlines():
                if "gap" in line.lower() or "hl-gap" in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0.0 < val < 30.0: return val
                        except ValueError: pass
            return None
            
        except Exception:
            return None

    def evaluate(self, sequence: list, target_raw: float, sigma: float) -> float:
        """Evaluate sequence using an oligomeric 1/N scaling law extrapolation over g-xTB calculations."""
        pselfies = self.tokenizer.decode(sequence, skip_special_tokens=True) 

        # Initialize fallback strings immediately to shield the except block from NameErrors
        psmiles_raw = "failed_parse"
        smiles_raw = "failed_parse"
        smiles_clean = ""
        
        psmiles = self.pselfies_to_psmiles(pselfies)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        file_exists = os.path.isfile(self.log_path)

        if not psmiles:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                writer.writerow([pselfies, "", "", False, -1.0])
            return -1.0

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                psmiles_raw = pselfies
                smiles_raw = psmiles  # Snapshot the true pristine output structure containing [At] handles
                
                # ─── MONOMER CAPPING PARADIGM ───
                # Replace wildcards with Hydrogen exclusively for standalone monomer validation
                smiles_clean = psmiles.replace("[At]", "[H]").replace("At", "[H]").replace("[*]", "[H]").replace("*", "[H]").strip()
                smiles_clean = re.sub(r'\(\s*\)', '', smiles_clean)  
                smiles_clean = smiles_clean.replace("HH", "H").strip()

                mol_check = Chem.MolFromSmiles(smiles_clean) if smiles_clean else None

                if mol_check is not None:
                    try:
                        Chem.SanitizeMol(mol_check)
                    except Exception:
                        mol_check = None
                
                # CRITICAL VERIFICATION: Validate monomer graph rules before running xTB
                # Open, evaluate, and write the sequence log state cleanly in a single unbroken pass
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not smiles_clean or mol_check is None:
                        writer.writerow([psmiles_raw, smiles_raw, smiles_clean, False, -1.0])
                        return -1.0
                    else:
                        writer.writerow([psmiles_raw, smiles_raw, smiles_clean, True, "pending"])

                mol = Chem.MolFromSmiles(smiles_clean)

                # Element Allowlist Filter
                allowed_atoms = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'B', 'H'}
                mol_atoms = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                if not mol_atoms.issubset(allowed_atoms):
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.9])
                    return -0.9
            
                heavy_atoms = mol.GetNumHeavyAtoms()
                if heavy_atoms < 6 or heavy_atoms > 90:
                    with open(self.log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                        writer.writerow([psmiles_raw, smiles_raw, smiles_clean, False, -0.8])
                    return -0.8

                self.true_oracle_calls += 2  
                    
                monomer_smiles = smiles_clean

                # ─── LITERATURE REACTION SMARTS TRIMER LINKER ───
                try:
                    m1 = Chem.MolFromSmiles(smiles_raw)
                    m2 = Chem.MolFromSmiles(smiles_raw)
                    m3 = Chem.MolFromSmiles(smiles_raw)
                    
                    # Match heavy atoms bound to Astatine wildcards, drop them, and weld backbones natively
                    rxn = AllChem.ReactionFromSmarts('([*:1]-[At]).([*:2]-[At]) >> [*:1]-[*:2]')
                    
                    # Step A: Link Monomer 1 + Monomer 2 into a Dimer
                    products1 = rxn.RunReactants((m1, m2))
                    if not products1:
                        with open(self.log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                        return -0.7
                        
                    dimer = products1[0][0]
                    try:
                        # FIX: Enforce intermediate sanitization to preserve clean valence rules
                        Chem.SanitizeMol(dimer)
                    except Exception:
                        with open(self.log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                        return -0.7
                    
                    # Step B: Link Dimer + Monomer 3 into a raw un-capped Trimer
                    products2 = rxn.RunReactants((dimer, m3))
                    if not products2:
                        with open(self.log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                        return -0.7
                        
                    raw_trimer = products2[0][0]
                    try:
                        Chem.SanitizeMol(raw_trimer)
                        raw_trimer_smiles = Chem.MolToSmiles(raw_trimer)
                        
                        # Swap terminal outer boundary handles out for stable Hydrogens
                        trimer_smiles = raw_trimer_smiles.replace("[At]", "[H]").replace("At", "[H]").replace("[*]", "[H]").replace("*", "[H]").strip()
                        
                        final_trimer_mol = Chem.MolFromSmiles(trimer_smiles)
                        if final_trimer_mol is None:
                            with open(self.log_path, 'a', newline='') as f:
                                csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                            return -0.7
                        Chem.SanitizeMol(final_trimer_mol)
                        trimer_smiles = Chem.MolToSmiles(final_trimer_mol)
                    except Exception:
                        with open(self.log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                        return -0.7
                except Exception:
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.7])
                    return -0.7
                # ──────────────────────────────────────────────── 

                # Calculate SA Score BEFORE xTB
                sa_score = sascorer.calculateScore(mol_check)

                # Removing SA return, sa_penalty_factor takes care of it
                # # If it's too complex, bypass xTB entirely and return a soft penalty
                # if sa_score > 4.5:
                #     with open(self.log_path, 'a', newline='') as f:
                #         csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.6])   
                #     return -0.6

                gap_n1 = self._execute_composite_calculation(tmpdir, monomer_smiles, "monomer")
                if gap_n1 is None or gap_n1 < 0.0 or gap_n1 > 30.0:
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.4])
                    return -0.4
                    
                gap_n3 = self._execute_composite_calculation(tmpdir, trimer_smiles, "trimer")
                if gap_n3 is None or gap_n3 < 0.0 or gap_n3 > 30.0:
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.4])
                    return -0.4

                infinite_gap = (3.0 * gap_n3 - gap_n1) / 2.0
                if not (0.0 < infinite_gap < 30.0):
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -0.4])
                    return -0.4

                # 1. Calculate the base physics reward (0 to 1)
                physics_reward = math.exp(-((infinite_gap - target_raw) ** 2) / (2 * sigma ** 2))
                
                # 2. Calculate the Synthetic Accessibility Score (1 to 10)
                # We use the monomer_smiles because that is the actual building block you would synthesize
                # sa_score = sascorer.calculateScore(mol_check) 
                
                # 3. Create a penalty scalar. 
                # If SA is 1 (easy), penalty is ~1.0 (no reduction). 
                # If SA is 10 (hard), penalty is ~0.1 (massive reduction).
                sa_penalty_factor = math.exp(-0.5 * max(0.0, sa_score - 3.0))

                # 4. Combine into a Multi-Objective Reward
                combined_reward = physics_reward * sa_penalty_factor

                # Convert to the [-1.0, 1.0] scale required by your critic head
                final_score = (combined_reward * 1.3) - 0.3

                # ---------------------------------------------------------
                # 5. Global Fingerprint Tracker & Exponential Reward Decay
                # ---------------------------------------------------------
                if not hasattr(self, 'global_structure_counts'):
                    self.global_structure_counts = {}
                    
                # Extract canonical SMILES to ensure identical graphs match
                canonical_smiles = Chem.MolToSmiles(mol_check, isomericSmiles=False)
                n = self.global_structure_counts.get(canonical_smiles, 0)
                alpha = 0.20 # Diversity penalty (20% decay per duplicate)
                
                if final_score > 0:
                    # Decay positive exploits exponentially toward zero
                    final_score = final_score * ((1.0 - alpha) ** n)
                # Removed negative reward death spiral
                    
                # Log the occurrence for future iterations
                self.global_structure_counts[canonical_smiles] = n + 1
                
                # Clamp to the -1.0 minimum boundary
                final_score = max(-1.0, final_score)
                # ---------------------------------------------------------
                
                with open(self.log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, final_score])
                return final_score

        except Exception:
            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([psmiles_raw, smiles_raw, "", False, -1.0])
            return -1.0

    def backpropagate(self, node: Node, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N
            node = node.parent

    def get_policy_target(self, root: Node) -> tuple:
        """Get MCTS policy from visit counts."""
        token_ids = list(root.children.keys())
        counts = torch.tensor([root.children[tid].N for tid in token_ids], dtype=torch.float32)

        # FIX: Safe probability calculation to prevent NaN crashes
        if counts.sum() == 0:
            probs = torch.tensor([root.children[tid].P for tid in token_ids], dtype=torch.float32)
            probs = probs / probs.sum()
        elif self.tau == 0:
            probs = torch.zeros_like(counts)
            probs[counts.argmax()] = 1.0
        else:
            counts_temp = counts ** (1.0 / self.tau)
            probs = counts_temp / counts_temp.sum()

        return torch.tensor(token_ids, dtype=torch.long), probs

    def run_episode(self, target_norm: float, sigma: float, batch_size: int = 8) -> list:
        """Run one Batched MCTS episode using raw physical targets throughout tree lifecycle."""
        root = Node(
            token_id=self.tokenizer.bos_id,
            parent=None,
            children={},
            P=1.0,
        )
        sequence = [self.tokenizer.bos_id]
        trajectory = []

        target_raw = target_norm * (self.target_max - self.target_min) + self.target_min
        active_simulations = 10 if self.true_oracle_calls < 30 else self.n_simulations

        for step in range(self.max_length - 1):
            num_batches = max(2, active_simulations // batch_size)
            
            for _ in range(num_batches):
                leaves = []
                leaf_seqs = []
                visited_this_batch = set()

                # --- PHASE 1: Traversal & Virtual Loss ---
                for _ in range(batch_size):
                    node = root
                    leaf_seq = []
                    
                    while node.children and node.token_id != self.tokenizer.eos_id:
                        best_score = -float('inf')
                        best_child = None
                        
                        for action, child in node.children.items():
                            if id(child) in visited_this_batch:
                                continue
                            
                            current_score = child.Q + self.c_puct * child.P * (math.sqrt(node.N) / (1 + child.N))
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_child = child
                                
                        if best_child is None:
                            best_child = max(node.children.values(), key=lambda c: c.Q)
                            
                        node = best_child
                        node.N += 1
                        node.W -= 1.0
                        node.Q = node.W / node.N

                    curr_trace = node
                    while curr_trace is not None:
                        leaf_seq.append(curr_trace.token_id)
                        curr_trace = curr_trace.parent
                    
                    leaf_seq = sequence[:-1] + list(reversed(leaf_seq))
                    
                    visited_this_batch.add(id(node))
                    leaves.append(node)
                    leaf_seqs.append(leaf_seq)

                # --- PHASE 2: Batched Network Evaluation ---
                batched_tensor = []
                for seq in leaf_seqs:
                    padded = seq + [self.tokenizer.pad_id] * (self.max_length - len(seq))
                    batched_tensor.append(padded[:self.max_length])
                    
                seq_tensor = torch.tensor(batched_tensor, dtype=torch.long, device=self.device)
                target_tensor = torch.tensor([target_raw] * batch_size, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    logits_batch, values_batch = self.policy(seq_tensor, property=target_tensor)

                # --- PHASE 3: Expansion & Backprop ---
                for i, leaf in enumerate(leaves):
                    reward = values_batch[i, len(leaf_seqs[i]) - 1].item()
                    
                    if leaf.token_id != self.tokenizer.eos_id and len(leaf_seqs[i]) < self.max_length:
                        self.expand(leaf, leaf_seqs[i], target_raw) 
                        
                    # Backpropagate: Cleanly separate Virtual Loss Undo from Real Updates
                    curr = leaf
                    while curr is not None:
                        if curr != root:
                            # 1. Undo the fake virtual loss penalties
                            curr.N -= 1
                            curr.W += 1.0
                            
                        # 2. Apply the true physical rewards and visit counts
                        curr.N += 1
                        curr.W += reward
                        curr.Q = curr.W / curr.N
                        
                        curr = curr.parent

            # Get MCTS policy and step forward
            token_ids, mcts_probs = self.get_policy_target(root)
            sampled_idx = torch.multinomial(mcts_probs, 1).item()
            next_token_id = token_ids[sampled_idx].item()

            trajectory.append((sequence.copy(), token_ids, mcts_probs, target_raw))
            sequence.append(next_token_id)

            if next_token_id in root.children:
                root = root.children[next_token_id]
                root.parent = None
            else:
                root = Node(token_id=next_token_id, parent=None, children={}, P=1.0)

            if next_token_id == self.tokenizer.eos_id:
                break

        final_reward = self.evaluate(sequence, target_raw, sigma)
        clipped_critic_reward = final_reward # No clipping! Huber loss handles outliers

        updated_trajectory = []
        T = len(trajectory)
        gamma = 1.0
        for t, step_data in enumerate(trajectory):
            discounted_reward = (gamma ** (T - 1 - t)) * final_reward
            discounted_value_target = (gamma ** (T - 1 - t)) * clipped_critic_reward
            updated_trajectory.append((*step_data, discounted_reward, discounted_value_target))

        return updated_trajectory


class MCTSFineTuner:
    """Trainer for MCTS-based finetuning."""

    def __init__(
        self,
        policy_model: PSELFIESLanguageModel,
        reference_model: PSELFIESLanguageModel,
        tokenizer: PSELFIESTokenizer,
        target_min: float,
        target_max: float,
        n_simulations: int = 200,
        kl_weight: float = 0.1,
        lr: float = 1e-4,
        home_dir: str = "./checkpoints/mcts"
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.tokenizer = tokenizer
        self.target_min = target_min
        self.target_max = target_max
        self.kl_weight = kl_weight
        self.device = next(policy_model.parameters()).device
        

        self.mcts = MCTSSearch(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            target_min=target_min,
            target_max=target_max,
            n_simulations=n_simulations,
            save_dir=home_dir,
        )

        # Only optimize AdaLN, output head, and value head parameters that require grad
        trainable_params = (
            list(policy_model.adaln_property.parameters()) +
            list(policy_model.property_fourier.parameters()) +
            list(policy_model.output_head.parameters()) +
            list(policy_model.value_head.parameters())
        )
        self.trainable_params = [p for p in trainable_params if p.requires_grad]

        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=0.01)

        self.replay_buffer = deque(maxlen=10_000)
        self.min_buffer_size = 512   # don't train until buffer has this many samples
        self.batch_size = 512
        self.scaler = torch.amp.GradScaler('cuda')
        self.iteration = 0
        self.kl_beta = 0.50  # Initialize adaptive KL penalty weight

    def train_iteration(self, target: float, sigma: float) -> dict:
        """Run one MCTS episode, store in buffer, train on static compilation-friendly batch sizes."""
        # ── 1. Run episode and push steps into replay buffer ──────────
        episode_data = self.mcts.run_episode(target, sigma)
        self.replay_buffer.extend(episode_data)

        # ── 2. Wait until buffer is large enough ──────────────────────
        if len(self.replay_buffer) < self.min_buffer_size:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl_loss": 0.0,
                "buffer_size": len(self.replay_buffer),
                "value_loss": 0.0
            }

        # ── 3. Sample a random mini-batch from the buffer ─────────────
        batch = random.sample(self.replay_buffer, self.batch_size)

        # ── 4. Collate sequences ──────────────────────────────────────
        sequences     = [item[0] for item in batch]
        token_ids     = [item[1] for item in batch]
        mcts_probs    = [item[2] for item in batch]
        targets       = [item[3] for item in batch]
        final_rewards = [item[4] for item in batch]
        value_targets = [item[5] for item in batch]

        pad_id = self.tokenizer.pad_id

        # Always pad to the global max space tracking limit to maintain static tensor shapes
        padded_seqs = []
        clipped_lengths = []
        for seq in sequences:
            seq = seq[:self.tokenizer.max_length]  # clip first
            clipped_lengths.append(len(seq))
            padded = seq + [pad_id] * (self.tokenizer.max_length - len(seq))
            padded_seqs.append(padded)
 
        # Move all input arrays safely onto device memory
        seq_tensor           = torch.tensor(padded_seqs, dtype=torch.long, device=self.device)
        target_tensor        = torch.tensor(targets, dtype=torch.float32, device=self.device)
        true_rewards_tensor  = torch.tensor(final_rewards, dtype=torch.float32, device=self.device)
        value_targets_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.device)

        # ─── FIXED 1 & 2: REORDERED AND ALIGNED ADVANTAGE NORMALIZATION ───
        reward_mean = true_rewards_tensor.mean()
        reward_std = true_rewards_tensor.std()
        
        if reward_std < 1e-6:
            # FIX: Use zeros_like so we DO NOT reinforce garbage sequences. 
            # A gradient of 0 means "skip updating on this trash batch".
            advantages = torch.zeros_like(true_rewards_tensor)
        else:
            # Standard RL Advantage tracking scaled relative to the batch baseline
            advantages = (true_rewards_tensor - reward_mean) / (reward_std + 1e-8)
            advantages = advantages.clamp(-3.0, 3.0)  
        # ──────────────────────────────────────────────────────────────────

        seq_lengths = torch.tensor(clipped_lengths, dtype=torch.long, device=self.device)
        batch_indices = torch.arange(self.batch_size, device=self.device)

        # ── 5. Single batched reference forward (no grad + modern autocast) ───
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ref_logits, _ = self.reference(seq_tensor, property=target_tensor)
            ref_log_probs = F.log_softmax(ref_logits[batch_indices, seq_lengths - 1, :].float(), dim=-1)

        # ── 6. Single batched policy forward (modern autocast) ────────
        with torch.amp.autocast('cuda'):
            policy_logits, policy_values = self.policy(seq_tensor, property=target_tensor)

        log_probs = F.log_softmax(policy_logits[batch_indices, seq_lengths - 1, :].float(), dim=-1)

        # ── 7. Build sparse MCTS target over full vocab ─────────────
        mcts_probs_tensor = torch.zeros(
            self.batch_size, self.tokenizer.vocab_size, device=self.device
        )
        for i, (tids, probs) in enumerate(zip(token_ids, mcts_probs)):
            if isinstance(tids, torch.Tensor):
                tids = tids.to(self.device)
            else:
                tids = torch.tensor(tids, dtype=torch.long, device=self.device)
            if isinstance(probs, torch.Tensor):
                probs = probs.to(self.device)
            else:
                probs = torch.tensor(probs, dtype=torch.float32, device=self.device)
            mcts_probs_tensor[i, tids] = probs

        # ── 8. Vectorized losses ────────────────────────────────────
        cross_entropy_loss = -(mcts_probs_tensor * log_probs).sum(dim=-1)
        
        # FIX 3: Multiply Cross-Entropy loss by advantages to enable Reinforcement Learning updates
        policy_loss = (cross_entropy_loss * advantages).mean()
        
        probs_pol   = log_probs.exp()
        
        kl_loss          = (probs_pol * (log_probs - ref_log_probs)).sum(dim=-1).mean()

        extracted_values = policy_values[batch_indices, seq_lengths - 1].view(-1).float()
        
        # FIX 4: Track the value head loss cleanly against the correct value targets tensor
        value_loss = F.huber_loss(extracted_values, value_targets_tensor.view(-1), reduction='none', delta=0.5).mean()

        positive_rewards_count = sum(1 for item in self.replay_buffer if item[4] > 0.0)
        
        # 3. Adaptive KL Controller
        target_kl = 0.10
        if kl_loss.item() > target_kl * 1.5:
            self.kl_beta = min(self.kl_beta * 1.5, 2.0)
        elif kl_loss.item() < target_kl * 0.5:
            self.kl_beta = max(self.kl_beta * 0.7, 0.01)

        # Combine Losses
        loss = policy_loss + self.kl_beta * kl_loss + value_loss
        # Previous Loss (for reference): loss = policy_loss + self.kl_weight * kl + value_loss

        # ── 9. Single backward + optimizer step (AMP scaler) ─────────
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # --- NEW: GRADIENT NORM TRACKING ---
        total_norm = 0.0
        for p in self.trainable_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            max_norm=1.0,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.iteration += 1
        return {
            "loss":         loss.item(),
            "policy_loss":  policy_loss.item(),
            "kl_loss":      kl_loss.item(),
            "buffer_size":  len(self.replay_buffer),
            "value_loss":   value_loss.item(),
            "grad_norm":    total_norm,
            "reward_mean":  reward_mean.item(),
            "reward_std":   reward_std.item()
        }

def get_current_sigma(iteration):
    """Linearly anneals sigma from SIGMA_START to SIGMA_END between anneal iterations."""
    if iteration < SIGMA_ANNEAL_START:
        return SIGMA_START
    if iteration >= SIGMA_ANNEAL_END:
        return SIGMA_END
    progress = (iteration - SIGMA_ANNEAL_START) / (SIGMA_ANNEAL_END - SIGMA_ANNEAL_START)
    return SIGMA_START + progress * (SIGMA_END - SIGMA_START)

def seed_ar_replay_buffer(csv_path, tokenizer, replay_buffer, target_samples=1500, max_length=128):
    """Parses Khazana database polymer targets and populates the AR replay buffer
    with perfectly evaluated sequence trajectories."""
    print(f"\n📥 Pre-seeding AR replay buffer from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"⚠️ Target dataset path '{csv_path}' not found. Skipping buffer pre-seed loop.")
        return

    df = pd.read_csv(csv_path)
    pselfies_col = 'pselfies' if 'pselfies' in df.columns else df.columns[2]
    egc_col = 'Egc' if 'Egc' in df.columns else df.columns[1]
    
    if 'sa_score' not in df.columns:
        raise ValueError(f"CRITICAL: 'sa_score' column not found in {csv_path}. Please use Egc_pselfies_sa.csv!")
    sa_col = 'sa_score' if 'sa_score' in df.columns else df.columns[3]

    sampled_records = df.sample(n=min(target_samples, len(df)), random_state=42).reset_index(drop=True)
    
    for _, row in sampled_records.iterrows():
        pselfies_str = str(row[pselfies_col])
        true_gap = float(row[egc_col])
        sa_score = float(row[sa_col])
        
        tokens = tokenizer.encode(pselfies_str, add_special_tokens=True)
        if len(tokens) > max_length:
            continue
            
        target_raw = random.uniform(0.0, 10.0)
        SEED_SIGMA = 2.0
        
        # 1. Physics Reward
        physics_reward = math.exp(-((true_gap - target_raw) ** 2) / (2 * SEED_SIGMA ** 2))
        
        # 2. SA Penalty (Matched to evaluate_terminal continuous formula)
        sa_penalty_factor = math.exp(-0.5 * max(0.0, sa_score - 3.0))
        combined_reward = physics_reward * sa_penalty_factor
        
        # 3. Final RL Reward (Matches MCTS scale of [-0.3, 1.0])
        final_reward = (combined_reward * 1.3) - 0.3

        # Autoregressive MCTS expects a trajectory of step-by-step token generation
        sequence = [tokenizer.bos_id]
        gamma = 1.0
        T = len(tokens) - 1 # Exclude the initial BOS token
        
        for t, next_token_id in enumerate(tokens[1:]): # Loop through sequence starting after BOS
            # The MCTS buffer expects: (sequence, token_ids, mcts_probs, target_raw, discounted_reward, discounted_value)
            
            # Create a "fake" MCTS probability distribution where it was 100% confident in the true token
            token_ids = torch.tensor([next_token_id], dtype=torch.long)
            mcts_probs = torch.tensor([1.0], dtype=torch.float32)
            
            discounted_reward = (gamma ** (T - 1 - t)) * final_reward
            discounted_value_target = (gamma ** (T - 1 - t)) * final_reward
            
            replay_buffer.append((
                sequence.copy(), 
                token_ids, 
                mcts_probs, 
                target_raw, 
                discounted_reward, 
                discounted_value_target
            ))
            
            sequence.append(next_token_id)
            
    print(f"✓ Initialization Complete. AR Replay Buffer seeded with {len(replay_buffer)} trajectory steps.")

if __name__ == "__main__":
    # ── Config ──────────────────────────────────────────────────────
    class args:
        pretrain_ckpt  = "checkpoints/offline_finetune/best_offline_surrogate.pt"
        sft_ckpt       = "checkpoints/sft_ar/best_sft_model_ar.pt"
        target_min     = 0.0
        target_max     = 10.0
        iterations     = 5000
        n_simulations  = 50
        kl_weight      = 0.1
        lr             = 1e-4
        # Split Tracking Parameters
        SCRATCH_DIR      = "/storage/home/hcoda1/2/vyadav68/scratch/polymers/mcts_transformer/checkpoints/mcts"
        HOME_DIR         = "./checkpoints/mcts" # Base project tracking path

        save_every     = 100
        USE_COMPILE    = True  # Toggle flag for fullgraph compilation tracking
    # ────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = PSELFIESTokenizer.load("checkpoints/pretrain/tokenizer.pt")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load pretrained policy
    policy_model = PSELFIESLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
    ).to(device)

    ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]

    # Automatically strip the '_orig_mod.' compilation prefix if it exists
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    policy_model.load_state_dict(state_dict)
    print("Loaded pretrained policy")

    # Create frozen reference
    reference_model = PSELFIESLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
    ).to(device)
    
    # CRITICAL FIX: Load the Stage 2 SFT checkpoint as the KL reference, NOT the pretrain checkpoint.
    sft_ckpt = torch.load(args.sft_ckpt, map_location=device)
    sft_state = sft_ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in sft_state.keys()):
        sft_state = {k.replace('_orig_mod.', ''): v for k, v in sft_state.items()}
    reference_model.load_state_dict(sft_state)

    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    print("Created frozen reference model")

    # ── COMPILER ENVIRONMENT PATCHES ────────────────────────────────
    import torch._dynamo
    import transformers.utils.generic as hf_generic
    import transformers.utils.output_capturing as hf_capture

    # Direct Graph block permission injection
    torch._dynamo.allow_in_graph(hf_generic.ContextManagers)

    # Bind explicit global instances across environment scope drops
    hf_generic.torch = torch
    hf_capture.torch = torch
    # ────────────────────────────────────────────────────────────────

    # Initialize trainer, passing SCRATCH_DIR for high-frequency runtime operations
    trainer = MCTSFineTuner(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        target_min=args.target_min,
        target_max=args.target_max,
        n_simulations=args.n_simulations,
        kl_weight=args.kl_weight,
        lr=args.lr,
        home_dir=args.HOME_DIR,    # <-- Route checkpoints and long-term logs to home
    )

    # ── COMPILATION ROUTINE ─────────────────────────────────────────
    if args.USE_COMPILE:
        try:
            print("\nCompiling policy and reference models with dynamic batch footprints...")
            # Set dynamic=True to permit rapid switching between batch sizes 1 and 512
            policy_model = torch.compile(policy_model, mode='default', fullgraph=False, dynamic=True)
            reference_model = torch.compile(reference_model, mode='default', fullgraph=False, dynamic=True)
            
            trainer.policy = policy_model
            trainer.reference = reference_model
            trainer.mcts.policy = policy_model
            trainer.mcts.reference = reference_model
            
            with torch.no_grad():
                # Warm up shape 1: Single-sequence MCTS lookahead step (Batch size = 1)
                search_seqs = torch.randint(0, tokenizer.vocab_size, (1, tokenizer.max_length), device=device)
                search_targets = torch.rand(1, device=device)
                _ = policy_model(search_seqs, property=search_targets)
                
                # Warm up shape 2: Batched SGD updates (Batch size = 512)
                batch_seqs = torch.randint(0, tokenizer.vocab_size, (512, tokenizer.max_length), device=device)
                batch_targets = torch.rand(512, device=device)
                _ = policy_model(batch_seqs, property=batch_targets)
                _ = reference_model(batch_seqs, property=batch_targets)
            print("✓ Both models compiled and cached successfully with flexible graphs!")
        except Exception as e:

            print(f"⚠ Could not compile models: {e}")
            traceback.print_exc()
            args.USE_COMPILE = False

    # Ensure both physical paths exist before the training loop kicks off
    os.makedirs(args.SCRATCH_DIR, exist_ok=True)
    os.makedirs(args.HOME_DIR, exist_ok=True)

    # ─── FRESH RUN LOG SANITIZATION PASSTHROUGH ───
    # Dynamically delete old transformer log assets if not resuming a preempted job
    if RESUME_FROM_ITER is None:
        print("Wiping old transformer fine-tuning logs for a clean tracking run...")
        for log_file in ["sequence_log.csv", "ar_sample_efficiency_metrics.csv"]:
            target_path = os.path.join(args.HOME_DIR, log_file)
            if os.path.exists(target_path):
                os.remove(target_path)
            # Secondary check just in case the logging path was routed directly to scratch
            scratch_target_path = os.path.join(args.SCRATCH_DIR, log_file)
            if os.path.exists(scratch_target_path):
                os.remove(scratch_target_path)
    # ──────────────────────────────────────────────

    start_iteration = 0

    if RESUME_FROM_ITER is not None:
        resume_path = os.path.join(args.SCRATCH_DIR, f"mcts_iter_{RESUME_FROM_ITER}.pt")
        if os.path.exists(resume_path):
            print(f"🔄 Resuming MCTS Finetuning from scratch checkpoint: {resume_path}")
            checkpoint_data = torch.load(resume_path, map_location=device)
            
            # Unpack weights safely regardless of torch.compile state wrappers
            raw_state_dict = checkpoint_data["model_state_dict"]
            
            # LINTER FIX: Iterate directly over the dict instead of raw_state_dict.keys()
            if any(k.startswith('_orig_mod.') for k in raw_state_dict) and not args.USE_COMPILE:
                raw_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
            
            policy_model.load_state_dict(raw_state_dict)
            start_iteration = checkpoint_data.get("iteration", RESUME_FROM_ITER)
            
            # FIX: Dynamically extend the execution range to accommodate the resume point
            args.iterations = start_iteration + 5000 
        else:
            print(f"⚠️ Target checkpoint '{resume_path}' not found. Booting fresh weights.")

    # Seed the buffer so Iteration 1 starts with a fully trained network!
    seed_ar_replay_buffer(
        csv_path="/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/Egc_pselfies_sa.csv", # Update to your exact path
        tokenizer=tokenizer,
        replay_buffer=trainer.replay_buffer,
        target_samples=1500,
        max_length=tokenizer.max_length
    )

    print(f"🚀 Starting fine-tuning execution loop at iteration {start_iteration + 1}")
    pbar = tqdm(range(start_iteration, args.iterations), desc="MCTS Finetuning")

    # Training loop
    for iteration in pbar:
        # Sample target in original scale, then normalize
        target_raw = random.uniform(args.target_min, args.target_max)
        
        target_norm = (target_raw - args.target_min) / (args.target_max - args.target_min)
        current_sigma = get_current_sigma(iteration)

        metrics = trainer.train_iteration(target_norm, current_sigma)

        if (iteration + 1) % 10 == 0:
            # 1. Calculate the rolling 100-step history metrics
            avg_reward = sum(
                item[4] for item in list(trainer.replay_buffer)[-100:]
            ) / min(100, len(trainer.replay_buffer))    

            rewards_last100 = [item[4] for item in list(trainer.replay_buffer)[-100:]]
            rolling_reward_std = (sum((r - avg_reward)**2 for r in rewards_last100) / max(len(rewards_last100), 1)) ** 0.5

            # 2. Safely pull the batch-specific metrics from the dictionary
            current_grad_norm = metrics.get('grad_norm', 0.0)
            batch_reward_mean = metrics.get('reward_mean', 0.0)
            batch_reward_std = metrics.get('reward_std', 0.0)

            # 3. Print cleanly without .item() errors
            print(
                f"Iter {iteration+1}/{args.iterations} | "
                f"sigma={current_sigma:.4f} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"grad_norm={current_grad_norm:.4f} "      
                f"batch_reward_mean={batch_reward_mean:.3f} "      
                f"batch_reward_std={batch_reward_std:.3f} "        
                f"kl={metrics['kl_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"avg_reward(last100)={avg_reward:.3f} "
                f"rolling_reward_std={rolling_reward_std:.3f} "
                f"target={target_raw:.3f} "
                f"buffer={metrics['buffer_size']} "
                f"xtb_calls={trainer.mcts.true_oracle_calls}"
            )
            
            # Export sample efficiency tracking parameters to permanent HOME directory
            log_path = f"{args.HOME_DIR}/ar_sample_efficiency_metrics.csv"
            new_row = pd.DataFrame([{
                "iteration":        iteration + 1,
                "loss":             metrics['loss'],
                "policy_loss":      metrics['policy_loss'],
                "kl_loss":          metrics['kl_loss'],
                "value_loss":       metrics['value_loss'],
                "avg_reward":       avg_reward,
                "target":           target_raw,
                "buffer_size":      metrics['buffer_size'],
                "xtb_calls":        trainer.mcts.true_oracle_calls,
            }])
            if os.path.exists(log_path):
                new_row.to_csv(log_path, mode='a', header=False, index=False)
            else:
                new_row.to_csv(log_path, mode='w', header=True, index=False)

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'policy': f"{metrics['policy_loss']:.4f}",
                'kl': f"{metrics['kl_loss']:.4f}",
                'value': f"{metrics['value_loss']:.4f}",
                'avg_reward': f"{avg_reward:.3f}",
                'sigma': f"{current_sigma:.3f}",
                'xtb': trainer.mcts.true_oracle_calls,
                'buffer': metrics['buffer_size']
            })

        if (iteration + 1) % args.save_every == 0:
            # Route intermediate weights strictly to high-performance scratch space
            torch.save({
                "model_state_dict": policy_model.state_dict(),
                "iteration": iteration + 1,
                "target_min": args.target_min,
                "target_max": args.target_max,
            }, f"{args.SCRATCH_DIR}/mcts_iter_{iteration+1}.pt")
            print(f"Saved intermediate checkpoint to scratch at iteration {iteration+1}")

    print("MCTS finetuning complete!")

    # ── SAVE FINAL TARGET ARTIFACTS TO HOME ────────────────────────
    final_checkpoint_path = f"{args.HOME_DIR}/final_mcts_ar_policy.pt"
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "iteration": args.iterations,
        "target_min": args.target_min,
        "target_max": args.target_max,
    }, final_checkpoint_path)
    
    # Mirror copy the tokenizer into home beside it for immediate pipeline serving
    tokenizer.save(f"{args.HOME_DIR}/tokenizer.pt")
    print(f"🎉 Production-ready fine-tuned model and tokenizer archived safely in home: {final_checkpoint_path}")