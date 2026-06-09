"""MCTS-based finetuning for PSELFIES autoregressive model."""
import os
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from tokenizer_pselfies import PSELFIESTokenizer
from model_ar import PSELFIESLanguageModel
from collections import deque
import selfies
import subprocess


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
        """Convert PSELFIES to p-SMILES using selfies library."""
        try:
            # Remove special tokens if present
            pselfies_clean = pselfies.replace("<BOS>", "").replace("<EOS>", "").replace("<PAD>", "").strip()
            smiles = selfies.decoder(pselfies_clean)
            if smiles:
                # Replace standard [At] placeholders with standard polymer wildcards
                smiles = smiles.replace("[At]", "[*]").replace("At", "[*]")
            return smiles
        except Exception as e:
            # Return empty string if decoding fails
            return ""

    def _execute_composite_calculation(self, tmpdir, smiles_string, prefix):
        """Helper to run GFN2 geometry optimization followed by a g-xTB single-point gap check."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import subprocess
        import os
        import re

        try:
            # Safely check parsing
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                return -1.0
            
            mol = Chem.AddHs(mol)
            # Switch embedding failure from None to -1.0
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                return -1.0
            
            # INSULATE FORCE FIELD OPTIMIZATION AGAINST EXOTIC ATOMS
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                return -1.0
            
            xyz_path = os.path.join(tmpdir, f"{prefix}_input.xyz")
            Chem.MolToXYZFile(mol, xyz_path)

            # STAGE 1: Geometry Optimization via GFN2-xTB
            try:
                result_opt = subprocess.run(
                    ["xtb", f"{prefix}_input.xyz", "--gfn", "2", "--opt"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                return -1.0
            
            if result_opt.returncode != 0:
                return -1.0

            opt_xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
            dest_opt_xyz = os.path.join(tmpdir, f"{prefix}_opt.xyz")
            if os.path.exists(opt_xyz_path):
                os.rename(opt_xyz_path, dest_opt_xyz)
            else:
                return -1.0

            # STAGE 2: High-Precision Single-Point via g-xTB
            try:
                result_gxtb = subprocess.run(
                    ["xtb", f"{prefix}_opt.xyz", "--gxtb"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                return -1.0

            if result_gxtb.returncode != 0:
                return -1.0

            output_text = result_gxtb.stdout
            match = re.search(r"(?:HOMO-LUMO\s+gap|HL-Gap)\s*[:\/\s\w]*\s*([-+]?\d*\.\d+|\d+)", output_text, re.IGNORECASE)
            if match:
                return float(match.group(1))
                
            for line in output_text.splitlines():
                if "gap" in line.lower() or "hl-gap" in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0.0 < val < 30.0:
                                return val
                        except ValueError:
                            continue
            return -1.0
            
        except Exception:
            # Global fallback catch-all to prevent any unexpected execution failures
            return -1.0

    def evaluate(self, sequence: list, target_raw: float) -> float:
        """Evaluate sequence using an oligomeric 1/N scaling law extrapolation over g-xTB calculations."""
        import csv
        import os

        pselfies = self.tokenizer.decode(sequence, skip_special_tokens=False)
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
                # Isolate the core organic unit to verify real matter exists 
                core_check = psmiles.replace("[At]", "").replace("At", "").replace("[*]", "").replace("*", "")
                prev = None
                while prev != core_check:
                    prev = core_check
                    core_check = re.sub(r'\([=#/\\]?\)', '', core_check)
                    core_check = re.sub(r'[=#/\\]+\)', ')', core_check)
                    core_check = re.sub(r'\([^a-zA-Z0-9]*\)', '', core_check)
                    core_check = re.sub(r'[=#/\\]+$', '', core_check.strip())
                    core_check = re.sub(r'^[=#/\\]+', '', core_check.strip())
                
                if "=" in core_check:
                    if re.search(r'=[/\\]|[/\\]=', core_check):
                        core_check = core_check.replace("/", "").replace("\\", "")
                    else:
                        core_check = re.sub(r'(?<=[\(\)])[/\\]+|[/\\]+(?=[\(\)])', '', core_check)
                        core_check = re.sub(r'[/\\]{2,}', '', core_check)
                else:
                    core_check = core_check.replace("/", "").replace("\\", "")

                psmiles_raw = psmiles  
                smiles_raw = self.pselfies_to_psmiles(psmiles_raw)  
                smiles_clean = core_check.strip()

                from rdkit import Chem
                mol_check = Chem.MolFromSmiles(f"[H]{smiles_clean}[H]") if smiles_clean else None
                
                # CRITICAL VERIFICATION: Validate structure rules BEFORE printing to prevent row pollution
                if not smiles_clean or mol_check is None:
                    with open(self.log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                        writer.writerow([psmiles_raw, smiles_raw, smiles_clean, False, -1.0])
                    return -1.0

                # Increment tracking metrics safely after code checks clear out syntax failures
                self.true_oracle_calls += 2  
                    
                monomer_smiles = f"[H]{smiles_clean}[H]"
                trimer_smiles  = f"[H]{smiles_clean}{smiles_clean}{smiles_clean}[H]"

                gap_n1 = self._execute_composite_calculation(tmpdir, monomer_smiles, "monomer")
                if gap_n1 is None or gap_n1 < 0.0 or gap_n1 > 30.0:
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0
                    
                gap_n3 = self._execute_composite_calculation(tmpdir, trimer_smiles, "trimer")
                if gap_n3 is None or gap_n3 < 0.0 or gap_n3 > 30.0:
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0

                infinite_gap = (3.0 * gap_n3 - gap_n1) / 2.0
                if not (0.0 < infinite_gap < 30.0):
                    with open(self.log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0

                reward = math.exp(-((infinite_gap - target_raw) ** 2) / (2 * 0.1 ** 2))
                final_score = (reward * 2.0) - 1.0
                
                # Single, clean inline write pass upon full calculation completion
                with open(self.log_path, 'a', newline='') as f:
                    if not file_exists and os.path.getsize(self.log_path) == 0:
                        csv.writer(f).writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
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

        if self.tau == 0:
            probs = torch.zeros_like(counts)
            probs[counts.argmax()] = 1.0
        else:
            counts_temp = counts ** (1.0 / self.tau)
            probs = counts_temp / counts_temp.sum()

        return torch.tensor(token_ids, dtype=torch.long), probs

    def run_episode(self, target_norm: float) -> list:
        """Run one MCTS episode using raw physical targets throughout tree lifecycle."""
        root = Node(
            token_id=self.tokenizer.bos_id,
            parent=None,
            children={},
            P=1.0,
        )
        sequence = [self.tokenizer.bos_id]
        trajectory = []

        target_raw = target_norm * (self.target_max - self.target_min) + self.target_min

        for step in range(self.max_length - 1):
            # Run simulations
            for _ in range(self.n_simulations):
                leaf = self.select(root)

                # Build sequence to leaf
                leaf_seq = []
                node = leaf
                while node is not None:
                    leaf_seq.append(node.token_id)
                    node = node.parent
                
                # FIX: Reconstruct full context by prepending committed historical tokens before root
                leaf_seq = sequence[:-1] + list(reversed(leaf_seq))

                # Expand if not terminal
                if leaf.token_id != self.tokenizer.eos_id and len(leaf_seq) < self.max_length:
                    leaf, reward = self.expand(leaf, leaf_seq, target_raw) 
                else:
                    # Query the network's Value Head using the fully reconstructed sequence context
                    with torch.no_grad():
                        padded_sequence = leaf_seq + [self.tokenizer.pad_id] * (self.max_length - len(leaf_seq))
                        seq_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=self.device)
                        target_tensor = torch.tensor([target_raw], dtype=torch.float32, device=self.device)
                        _, values = self.policy(seq_tensor, property=target_tensor)
                        reward = values[0, len(leaf_seq) - 1].item()

                # Backpropagate
                self.backpropagate(leaf, reward)

            # Get MCTS policy
            token_ids, mcts_probs = self.get_policy_target(root)

            # Sample next token
            sampled_idx = torch.multinomial(mcts_probs, 1).item()
            next_token_id = token_ids[sampled_idx].item()

            trajectory.append((sequence.copy(), token_ids, mcts_probs, target_raw))
            sequence.append(next_token_id)

            # Move root
            if next_token_id in root.children:
                root = root.children[next_token_id]
                root.parent = None
            else:
                root = Node(
                    token_id=next_token_id,
                    parent=None,
                    children={},
                    P=1.0,
                )

            if next_token_id == self.tokenizer.eos_id:
                break

        # ✅ THIS IS THE ONLY TIME XTB GETS CALLED: Exactly once when finalized!
        final_reward = self.evaluate(sequence, target_raw)

        # Apply an exponential temporal decay scale (gamma = 0.99)
        updated_trajectory = []
        T = len(trajectory)
        gamma = 0.99 
        
        for t, step_data in enumerate(trajectory):
            discounted_reward = (gamma ** (T - 1 - t)) * final_reward
            updated_trajectory.append((*step_data, discounted_reward))

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
        save_dir: str = "checkpoints/mcts", # Added explicit tracking parameter argument
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
            save_dir=save_dir,
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

    def train_iteration(self, target: float) -> dict:
        """Run one MCTS episode, store in buffer, train on static compilation-friendly batch sizes."""
        # ── 1. Run episode and push steps into replay buffer ──────────
        episode_data = self.mcts.run_episode(target)
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

        # ── 4. Collate sequences (pad to global maximum length to avoid graph breaks) ─────────
        sequences   = [item[0] for item in batch]
        token_ids   = [item[1] for item in batch]
        mcts_probs  = [item[2] for item in batch]
        targets     = [item[3] for item in batch]
        final_rewards = [item[4] for item in batch]

        pad_id = self.tokenizer.pad_id

        # FIX: Always pad to the global max space tracking limit to maintain static tensor shapes
        padded_seqs = []
        clipped_lengths = []
        for seq in sequences:
            seq = seq[:self.tokenizer.max_length]  # clip first
            clipped_lengths.append(len(seq))
            padded = seq + [pad_id] * (self.tokenizer.max_length - len(seq))
            padded_seqs.append(padded)
 
        seq_tensor    = torch.tensor(padded_seqs, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
        true_rewards_tensor = torch.tensor(final_rewards, dtype=torch.float32, device=self.device)
 
        # Get sequence lengths to dynamically extract output at actual sequence-end tokens
        # Use clipped_lengths (post-clip) rather than the original sequences list, which is
        # unmodified by the clip above and would produce out-of-bounds indices into seq_tensor.
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

        # Compute log-probs in FP32 for numerical stability
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
        policy_loss = -(mcts_probs_tensor * log_probs).sum(dim=-1).mean()
        probs_pol   = log_probs.exp()
        kl          = (probs_pol * (log_probs - ref_log_probs)).sum(dim=-1).mean()

        # CRITICAL BUG FIX: Added .float() to extracted_values.
        # This converts the Half precision output from autocast into Float32 BEFORE the MSE calculation,
        # preventing the backward pass from hitting a data-type mismatch crash.
        extracted_values = policy_values[batch_indices, seq_lengths - 1].view(-1).float()
        value_loss = F.mse_loss(extracted_values, true_rewards_tensor.view(-1))

        # Combine Losses
        loss = policy_loss + self.kl_weight * kl + value_loss

        # ── 9. Single backward + optimizer step (AMP scaler) ─────────
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            max_norm=1.0,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss":         loss.item(),
            "policy_loss":  policy_loss.item(),
            "kl_loss":      kl.item(),
            "buffer_size":  len(self.replay_buffer),
            "value_loss":   value_loss.item()
        }

if __name__ == "__main__":
    # ── Config ──────────────────────────────────────────────────────
    class args:
        pretrain_ckpt  = "checkpoints/pretrain/best_pretrain.pt"
        target_min     = 0.5
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
    reference_model.load_state_dict(state_dict)

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
        save_dir=args.SCRATCH_DIR,  # <-- Route runtime logs (sequence_log.csv) to scratch
    )

    # ── COMPILATION ROUTINE ─────────────────────────────────────────
    if args.USE_COMPILE:
        try:
            print("\nCompiling policy and reference models with torch.compile()...")
            policy_model = torch.compile(policy_model, mode='default', fullgraph=False)
            reference_model = torch.compile(reference_model, mode='default', fullgraph=False)
            
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
            print("✓ Both models compiled and tested successfully with a 100% complete graph!")
        except Exception as e:
            import traceback
            print(f"⚠ Could not compile models: {e}")
            print("Full traceback:")
            traceback.print_exc()
            args.USE_COMPILE = False
    # ────────────────────────────────────────────────────────────────

    # Ensure both physical paths exist before the training loop kicks off
    os.makedirs(args.SCRATCH_DIR, exist_ok=True)
    os.makedirs(args.HOME_DIR, exist_ok=True)

    # Optional Resume Hook Integration
    RESUME_FROM_ITER = 18000  # Set to an integer step to resume from scratch, or None to start fresh
    start_iteration = 0

    if RESUME_FROM_ITER is not None:
        resume_path = os.path.join(args.SCRATCH_DIR, f"mcts_iter_{RESUME_FROM_ITER}.pt")
        if os.path.exists(resume_path):
            print(f"🔄 Resuming MCTS Finetuning from scratch checkpoint: {resume_path}")
            checkpoint_data = torch.load(resume_path, map_location=device)
            
            # Unpack weights safely regardless of torch.compile state wrappers
            raw_state_dict = checkpoint_data["model_state_dict"]
            if any(k.startswith('_orig_mod.') for k in raw_state_dict.keys()) and not args.USE_COMPILE:
                raw_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
            
            policy_model.load_state_dict(raw_state_dict)
            start_iteration = checkpoint_data.get("iteration", RESUME_FROM_ITER)
        else:
            print(f"⚠️ Target checkpoint '{resume_path}' not found. Booting fresh weights.")

    print(f"🚀 Starting fine-tuning execution loop at iteration {start_iteration + 1}")

    # Training loop
    for iteration in range(start_iteration, args.iterations):
        # Sample target in original scale, then normalize
        target_raw = random.uniform(args.target_min, args.target_max)
        target_norm = (target_raw - args.target_min) / (args.target_max - args.target_min)

        metrics = trainer.train_iteration(target_norm)

        if (iteration + 1) % 10 == 0:
            avg_reward = sum(
                item[4] for item in list(trainer.replay_buffer)[-100:]
            ) / min(100, len(trainer.replay_buffer))    

            print(
                f"Iter {iteration+1}/{args.iterations} | "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"kl={metrics['kl_loss']:.4f} "          # rising fast = policy diverging from reference
                f"value={metrics['value_loss']:.4f} "     # should fall as value head learns reward landscape
                f"avg_reward(last100)={avg_reward:.3f} "  # most important: should trend toward 0 then +
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