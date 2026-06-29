import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm

from tokenizer_pselfies import PSELFIESTokenizer
from model_ar import PSELFIESLanguageModel 

# ============================================================================
# CONFIGURATION
# ============================================================================
PRETRAIN_CKPT = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_transformer/checkpoints/pretrain/best_pretrain.pt"
DATASET_PATH = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/Egc_pselfies_sa.csv" 
OUTPUT_DIR = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_transformer/checkpoints/sft_ar"

BATCH_SIZE = 64  
EPOCHS = 15
LEARNING_RATE = 1e-5 # Very small LR to gently adapt the backbone

TARGET_MIN = 0.0 
TARGET_MAX = 10.0 
# ============================================================================

class KhazanaSFTDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        
        self.sequences = []
        self.targets = []
        
        sequence_col = "pselfies" if "pselfies" in self.df.columns else self.df.columns[0]
        target_col = "Egc" if "Egc" in self.df.columns else self.df.columns[1]
        
        for sequence, bandgap in tqdm(zip(self.df[sequence_col], self.df[target_col]), total=len(self.df), desc="Tokenizing"):
            token_ids = self.tokenizer.encode(str(sequence), add_special_tokens=True)
            if len(token_ids) > self.tokenizer.max_length:
                continue
                
            self.sequences.append(token_ids)
            self.targets.append(float(bandgap))

        print(f"Loaded {len(self.sequences)} valid Khazana polymers for SFT.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = seq + [self.tokenizer.pad_id] * (self.tokenizer.max_length - len(seq))
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }

def train_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Starting Supervised Fine-Tuning (AR) on {device} (AMP: {use_amp})")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer directly from the original pretrain folder
    tokenizer = PSELFIESTokenizer.load("/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_transformer/checkpoints/pretrain/tokenizer.pt")
    
    model = PSELFIESLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
    ).to(device)

    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # UNFREEZE EVERYTHING FOR SFT
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    full_dataset = KhazanaSFTDataset(DATASET_PATH, tokenizer)
    
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.10 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # We ignore the pad token so it doesn't artificially inflate accuracy
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id) 
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            raw_targets = batch['target'].to(device)
            
            # Normalize target for the conditioning vector
            target_norm = ((raw_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)).clamp(0.0, 1.0)
            
            # Create standard AR inputs and labels (shift right by 1)
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Forward pass - we only care about the logits during SFT, ignore the value head
                logits, _ = model(x, property=target_norm)
                
                # Flatten for CrossEntropy
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'CE_Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                raw_targets = batch['target'].to(device)
                target_norm = ((raw_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)).clamp(0.0, 1.0)
                
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _ = model(x, property=target_norm)
                    loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                    total_val_loss += loss.item()
                    
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch+1} Results ---")
        print(f"Train NLL Loss: {avg_train_loss:.4f}")
        print(f"Val NLL Loss  : {avg_val_loss:.4f}\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save this as the base for the offline critic
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), 
                'val_loss': avg_val_loss,
            }, f"{OUTPUT_DIR}/best_sft_model_ar.pt")
            
if __name__ == "__main__":
    train_sft()