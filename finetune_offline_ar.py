import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import math
from tqdm import tqdm

from tokenizer_pselfies import PSELFIESTokenizer
from model_ar import PSELFIESLanguageModel  # Make sure this matches your AR model file

# ============================================================================
# CONFIGURATION
# ============================================================================
PRETRAIN_CKPT = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_transformer/checkpoints/sft_ar/best_sft_model_ar.pt"
DATASET_PATH = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/Egc_pselfies_sa.csv" 
OUTPUT_DIR = "./checkpoints/offline_finetune"

BATCH_SIZE = 64  # Dropped from 512 for the smaller 4.5k dataset
EPOCHS = 5
LEARNING_RATE = 1e-4

TARGET_MIN = 0.0  # Synced to true Egc bounds
TARGET_MAX = 10.0 # Synced to true Egc bounds
OFFLINE_SIGMA = 0.5 
# ============================================================================

class PolymerGenomeDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        
        self.sequences = []
        self.true_bandgaps = []
        self.sa_scores = [] # <-- NEW
        
        sequence_col = "pselfies" 
        target_col = "Egc" 
        sa_col = "sa_score" # <-- NEW
        
        for sequence, bandgap, sa in tqdm(zip(self.df[sequence_col], self.df[target_col], self.df[sa_col]), total=len(self.df)):
            token_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
            if len(token_ids) > self.tokenizer.max_length:
                continue
                
            self.sequences.append(token_ids)
            self.true_bandgaps.append(float(bandgap))
            self.sa_scores.append(float(sa)) # <-- NEW

        print(f"Loaded {len(self.sequences)} valid periodic polymers.")
        # ... (Keep the min/max safety assertions here) ...

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = seq + [self.tokenizer.pad_id] * (self.tokenizer.max_length - len(seq))
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.long),
            "true_bandgap": torch.tensor(self.true_bandgaps[idx], dtype=torch.float32),
            "sa_score": torch.tensor(self.sa_scores[idx], dtype=torch.float32) # <-- NEW
        }

def sanity_check(model, tokenizer, device):
    """Pre-flight check to guarantee AR tensor shapes won't silently corrupt training."""
    print("--- Running AR Architecture Sanity Check ---")
    model.eval()
    with torch.no_grad():
        dummy_ids = torch.randint(0, tokenizer.vocab_size, (4, tokenizer.max_length), device=device)
        dummy_prop = torch.rand(4, device=device)
        
        out = model(dummy_ids, property=dummy_prop)
        
        assert isinstance(out, tuple), "Model must return a tuple of (logits, values)"
        assert len(out) == 2, "Model must return exactly two elements"
        
        logits, values = out
        # AR models should output a value for EVERY token position: (B, L, 1) or (B, L)
        assert values.dim() in [2, 3], f"FATAL: Expected sequence values (B, L) or (B, L, 1), got {values.shape}."
    print("✓ Sanity Check Passed! Model outputs are safe.")

def train_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Training Causal AR Critic on {device} (AMP Enabled: {use_amp})")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = PSELFIESTokenizer.load("checkpoints/pretrain/tokenizer.pt")
    
    model = PSELFIESLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
    ).to(device)

    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # --- Pre-Flight Sanity Check ---
    sanity_check(model, tokenizer, device)

    # --- Robust Train/Eval Splitting ---
    model.eval() # Seal the transformer backbone dropouts
    for param in model.parameters():
        param.requires_grad = False
        
    trainable_params = []
    print("\n--- Active Trainable Parameters ---")
    for name, param in model.named_parameters():
        if any(key in name for key in ['value_head', 'adaln_property']):
            param.requires_grad = True
            trainable_params.append(param)
            print(f"  - {name}: {param.numel():,} params")

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total Trainable Parameters: {total_trainable:,}\n")
    assert total_trainable > 0, "FATAL: No trainable parameters found!"

    full_dataset = PolymerGenomeDataset(DATASET_PATH, tokenizer)
    
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)

    best_combined_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Explicitly freeze backbone dropouts, wake up heads
        model.eval() 
        for name, module in model.named_modules():
            if any(key in name for key in ['value_head', 'adaln_property']):
                module.train()
                
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            true_bandgaps = batch['true_bandgap'].to(device)
            
            mask = torch.rand_like(true_bandgaps) < 0.3
            # Match Gaussian sampling for RL smoothness
            local_targets = true_bandgaps + torch.randn_like(true_bandgaps) * 0.25
            global_targets = torch.rand_like(true_bandgaps) * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
            sampled_targets = torch.where(mask, local_targets, global_targets).clamp(TARGET_MIN, TARGET_MAX)
            
            # 1. Base Physics (0 to 1)
            physics_reward = torch.exp(-((true_bandgaps - sampled_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
            
            # 2. Extract SA Score from batch (only needed in the batch loop, define this once per loop)
            sa_scores = batch['sa_score'].to(device)
            
            # 3. Apply Multi-Objective SA Penalty (Matched to MCTS continuous formula)
            sa_penalty_factor = torch.exp(-0.5 * torch.clamp(sa_scores - 3.0, min=0.0))
            combined_reward = physics_reward * sa_penalty_factor
            
            # 4. Scale to continuous RL bounds (Matches MCTS scale of [-0.3, 1.0])
            expected_rl_reward = (combined_reward * 1.3) - 0.3
            target_norm = (sampled_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Clean AR forward pass (No timestep/attention mask needed)
                _, values = model(input_ids, property=target_norm)
                
                # AR-Specific [EOS] Extraction
                eos_mask = input_ids == tokenizer.eos_id
                seq_lengths = (input_ids != tokenizer.pad_id).sum(dim=1) - 1
                seq_lengths = seq_lengths.clamp(min=0)
                extraction_indices = torch.where(eos_mask.any(dim=1), eos_mask.float().argmax(dim=1), seq_lengths)
                
                batch_indices = torch.arange(input_ids.size(0), device=device)
                
                # Squeeze the final dimension safely
                predicted_values = values[batch_indices, extraction_indices]
                if predicted_values.dim() == 2 and predicted_values.shape[1] == 1:
                    predicted_values = predicted_values.squeeze(-1)
                
                # --- FATAL SAFETY CHECK 3: Shape Equality ---
                assert predicted_values.shape == expected_rl_reward.shape, f"FATAL Shape mismatch: {predicted_values.shape} vs {expected_rl_reward.shape}"
                
                loss = criterion(predicted_values, expected_rl_reward)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_perf_loss, total_med_loss, total_worst_loss = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                true_bandgaps = batch['true_bandgap'].to(device)
                
                eos_mask = input_ids == tokenizer.eos_id
                seq_lengths = (input_ids != tokenizer.pad_id).sum(dim=1) - 1
                seq_lengths = seq_lengths.clamp(min=0)
                extraction_indices = torch.where(eos_mask.any(dim=1), eos_mask.float().argmax(dim=1), seq_lengths)
                batch_indices = torch.arange(input_ids.size(0), device=device)

                # 1. PERFECT MATCH
                perf_targets = true_bandgaps.clone().clamp(TARGET_MIN, TARGET_MAX)
                perf_norm = (perf_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                perf_expected = torch.ones_like(true_bandgaps) 
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, perf_values = model(input_ids, property=perf_norm)
                    perf_preds = perf_values[batch_indices, extraction_indices]
                    if perf_preds.dim() == 2: perf_preds = perf_preds.squeeze(-1)
                    total_perf_loss += criterion(perf_preds, perf_expected).item()

                # 2. MEDIUM MATCH (+0.5 eV)
                med_targets = (true_bandgaps + OFFLINE_SIGMA).clamp(TARGET_MIN, TARGET_MAX)
                med_norm = (med_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                med_physics = torch.exp(-((true_bandgaps - med_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
                med_expected = (med_physics * 1.3) - 0.3
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, med_values = model(input_ids, property=med_norm)
                    med_preds = med_values[batch_indices, extraction_indices]
                    if med_preds.dim() == 2: med_preds = med_preds.squeeze(-1)
                    total_med_loss += criterion(med_preds, med_expected).item()
                
                # 3. WORST-CASE MISMATCH
                midpoint = (TARGET_MIN + TARGET_MAX) / 2.0
                worst_targets = torch.where(
                    true_bandgaps < midpoint,
                    torch.full_like(true_bandgaps, TARGET_MAX),
                    torch.full_like(true_bandgaps, TARGET_MIN)
                )
                worst_norm = (worst_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                worst_physics = torch.exp(-((true_bandgaps - worst_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
                worst_expected = ((worst_physics * 1.3) - 0.3)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, worst_values = model(input_ids, property=worst_norm)
                    worst_preds = worst_values[batch_indices, extraction_indices]
                    if worst_preds.dim() == 2: worst_preds = worst_preds.squeeze(-1)
                    total_worst_loss += criterion(worst_preds, worst_expected).item()
                
        avg_perf_loss = total_perf_loss / len(val_loader)
        avg_med_loss = total_med_loss / len(val_loader)
        avg_worst_loss = total_worst_loss / len(val_loader)
        
        # Strict validation checkpointing metric
        strict_val_loss = max(avg_perf_loss, avg_med_loss, avg_worst_loss)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train MSE    : {avg_train_loss:.4f}")
        print(f"Val Perfect  : {avg_perf_loss:.4f}")
        print(f"Val Medium   : {avg_med_loss:.4f}")
        print(f"Val Worst    : {avg_worst_loss:.4f}")
        print(f"Strict Val   : {strict_val_loss:.4f}\n")
        
        if strict_val_loss < best_combined_loss:
            best_combined_loss = strict_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'strict_val_loss': strict_val_loss,
                'target_min': TARGET_MIN,
                'target_max': TARGET_MAX,
                'offline_sigma': OFFLINE_SIGMA,
            }, f"{OUTPUT_DIR}/best_offline_surrogate.pt")
            
if __name__ == "__main__":
    train_offline()