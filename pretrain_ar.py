#!/usr/bin/env python
"""Pretraining script for PSELFIES autoregressive language model."""
import math
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from tokenizer_pselfies import PSELFIESTokenizer
from model_ar import PSELFIESLanguageModel


class PSELFIESTextDataset(Dataset):
    """Dataset for causal LM pretraining on PSELFIES strings."""

    def __init__(self, pselfies_list: list, tokenizer: PSELFIESTokenizer):
        self.pselfies_list = pselfies_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pselfies_list)

    def __getitem__(self, idx):
        pselfies = self.pselfies_list[idx]
        encoded = self.tokenizer.encode(pselfies, add_special_tokens=True)

        # Pad or truncate to max_length
        if len(encoded) < self.tokenizer.max_length:
            encoded = encoded + [self.tokenizer.pad_id] * (self.tokenizer.max_length - len(encoded))
        else:
            encoded = encoded[:self.tokenizer.max_length]

        return torch.tensor(encoded, dtype=torch.long)


def get_lr(step, warmup_steps, total_steps, base_lr):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


if __name__ == "__main__":
    # ── Config ──────────────────────────────────────────────────────
    class args:
        csv                  = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/PI1M_v2_pselfies.csv"
        epochs               = 10
        batch_size           = 256
        lr                   = 3e-4
        weight_decay = 0.01
        max_length   = 128
        # Split Tracking Parameters
        SCRATCH_DIR      = "/storage/home/hcoda1/2/vyadav68/scratch/polymers/mcts_transformer/checkpoints/pretrain"
        HOME_DIR         = "./checkpoints/pretrain" # Base project tracking path
        num_workers  = 4
        USE_COMPILE  = True  # Added toggle flag for fullgraph compilation tracking
        RESUME_FROM_STEP = 18000  # Set to None to disable resuming from a specific step
    # ────────────────────────────────────────────────────────────────

    os.makedirs(args.SCRATCH_DIR, exist_ok=True)
    os.makedirs(args.HOME_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.csv)
    col = "pselfies" if "pselfies" in df.columns else "pselfie"
    df = df.dropna(subset=[col])
    pselfies_list = df[col].tolist()
    print(f"Loaded {len(pselfies_list)} PSELFIES strings")

    # Build or load tokenizer
    tokenizer_path = os.path.join(args.HOME_DIR, "tokenizer.pt")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(args.SCRATCH_DIR, "tokenizer.pt")

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = PSELFIESTokenizer.load(tokenizer_path)
    else:
        print("Building tokenizer...")
        tokenizer = PSELFIESTokenizer(max_length=args.max_length)
        tokenizer.build_vocab(pselfies_list, min_freq=1)
        tokenizer.save(os.path.join(args.SCRATCH_DIR, "tokenizer.pt"))

    # Train/val split
    train_size = int(0.95 * len(pselfies_list))
    val_size = len(pselfies_list) - train_size
    train_data = pselfies_list[:train_size]
    val_data = pselfies_list[train_size:]

    train_ds = PSELFIESTextDataset(train_data, tokenizer)
    val_ds = PSELFIESTextDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
        drop_last=True,  # Ensure all batches are full for consistent training steps
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
    )

    # Model
    model = PSELFIESLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # ── COMPILATION ROUTINE ─────────────────────────────────────────
    if args.USE_COMPILE:
        try:
            print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode='default', fullgraph=True)
            
            with torch.no_grad():
                # Since Causal LM passes batch[:, :-1], the effective sequence length is max_length - 1
                dummy_inputs = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.max_length - 1), device=device)
                
                # FIX: Pass a live dummy tensor to force the compiler to trace the AdaLN conditioning graph
                dummy_properties = torch.rand(args.batch_size, device=device)
                _ = model(dummy_inputs, property=dummy_properties)
                
            print("✓ Model compiled and tested successfully with a 100% complete graph!")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}")
            args.USE_COMPILE = False
    # ────────────────────────────────────────────────────────────────

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── PLOTTING: HISTORY TRACKING ARCHIVE ──
    history = {
        "steps": [],
        "train_loss": [],
        "epochs": [],
        "val_loss": []
    }

    # Training setup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    SAVE_STEPS = 1000  # <--- Define your interval step size here

    # ── NEW LOADING ENGINE: CAUSAL TRANSFORMER RESUME ROUTINE ────────
    if args.RESUME_FROM_STEP is not None:
        # Route loading path strictly to scratch directory allocations
        checkpoint_path = os.path.join(args.SCRATCH_DIR, f"checkpoint-{args.RESUME_FROM_STEP}", "model.pt")
        if os.path.exists(checkpoint_path):
            print(f"\n🔄 Resuming Causal AR Transformer pretraining from step {args.RESUME_FROM_STEP}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract state allocations safely
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            global_step = checkpoint.get('global_step', args.RESUME_FROM_STEP)
            start_epoch = checkpoint.get('epoch', 0)
            history = checkpoint.get('history', history)
            print(f"✓ Resumed from step {global_step} | Restarting at Epoch {start_epoch + 1}")
        else:
            print(f"⚠️ Checkpoint path '{checkpoint_path}' not found! Booting fresh weights.")
    # ────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Causal LM: predict next token
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Adjust learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            # 1. SYNTHETIC PROPERTY GENERATION: Sample random properties on the fly
            current_batch_size = inputs.size(0)
            synthetic_properties = torch.rand(current_batch_size, device=device)
            
            # 2. CLASSIFIER-FREE GUIDANCE DROPOUT: Route 15% of inputs through [NULL] paths
            cfg_dropout_mask = torch.rand(current_batch_size, device=device) < 0.15
            synthetic_properties[cfg_dropout_mask] = -1.0  # Sentinel tracking value

            with torch.amp.autocast('cuda'):
                # Tuple unpacking: capture both logits and sequence values
                logits, sequence_values = model(inputs, property=synthetic_properties)
                
                # A. Main Causal Language Modeling Loss (Grammar)
                policy_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size),
                    targets.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
                
                # B. Value Head Anchor Loss (Stabilizes critic baseline at 0.0)
                value_loss = nn.functional.mse_loss(
                    sequence_values, torch.zeros_like(sequence_values)
                )
                
                # Joint objective combined scaling path
                loss = policy_loss + 0.1 * value_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1

            # Save step history metrics for tracking plots
            if global_step % 10 == 0:
                history["steps"].append(global_step)
                history["train_loss"].append(loss.item())
            
            pbar.set_postfix({"loss": loss.item(), "lr": lr})

            if global_step % SAVE_STEPS == 0:
                interval_path = os.path.join(args.SCRATCH_DIR, f"checkpoint-{global_step}")
                os.makedirs(interval_path, exist_ok=True)
                
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": {
                        "global_step": global_step,
                        "warmup_steps": warmup_steps,
                        "total_steps": total_steps
                    },
                    "tokenizer_vocab": tokenizer.vocab,
                    "vocab_size": tokenizer.vocab_size,
                    "max_length": tokenizer.max_length,
                    "epoch": epoch,  # Capture active training epoch index
                    "history": history
                }
                # Save the model parameters and the tokenizer right next to it
                torch.save(checkpoint, f"{interval_path}/model.pt")
                tokenizer.save(f"{interval_path}/tokenizer.pt")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                # Match training dimension configurations
                current_batch_size = inputs.size(0)
                synthetic_properties = torch.rand(current_batch_size, device=device)
                cfg_dropout_mask = torch.rand(current_batch_size, device=device) < 0.15
                synthetic_properties[cfg_dropout_mask] = -1.0

                with torch.amp.autocast('cuda'):
                    logits, sequence_values = model(inputs, property=synthetic_properties)
                    
                    policy_loss = nn.functional.cross_entropy(
                        logits.reshape(-1, tokenizer.vocab_size),
                        targets.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                    
                    value_loss = nn.functional.mse_loss(
                        sequence_values, torch.zeros_like(sequence_values)
                    )
                    
                    loss = policy_loss + 0.1 * value_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Record epoch-level history metrics
        history["epochs"].append(epoch + 1)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoints
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "tokenizer_vocab": tokenizer.vocab,
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.max_length,
            "epoch": epoch + 1,
        }

        # Save checkpoints to permanent Home directory project paths
        torch.save(checkpoint, f"{args.HOME_DIR}/last_pretrain.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, f"{args.HOME_DIR}/best_pretrain.pt")
            tokenizer.save(f"{args.HOME_DIR}/tokenizer.pt") # Ensure tokenizer is archived beside best model
            print(f"Saved best model to home (val_loss={val_loss:.4f})")
    
    try:
        print("\n📊 Generating optimization evaluation plots...")
        import matplotlib
        matplotlib.use('Agg')  # Suppress UI popups on remote cluster nodes
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(args.HOME_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Graph A: Step-Level Policy Loss Trend Curve
        ax1.plot(history["steps"], history["train_loss"], color="#1f77b4", alpha=0.4, label="Batch Loss")
        # Apply a running average window pass to make the graph clean and readable
        if len(history["train_loss"]) > 20:
            window = 20
            smooth_loss = pd.Series(history["train_loss"]).rolling(window=window, min_periods=1).mean()
            ax1.plot(history["steps"], smooth_loss, color="#d62728", linewidth=2, label="Smoothed Trend")
        ax1.set_title("Causal LM Pretraining Loss Over Steps", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Global Step Sequence", fontsize=10)
        ax1.set_ylabel("Cross Entropy + Value Loss Metric", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right")

        # Graph B: Validation Alignment Progress Curve
        ax2.plot(history["epochs"], history["val_loss"], color="#2ca02c", marker='o', linewidth=2, label="Validation Loss")
        ax2.set_title("Validation Convergence Profile", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch Number", fontsize=10)
        ax2.set_ylabel("Total Objective Loss", fontsize=10)
        ax2.set_xticks(history["epochs"])
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        chart_path = os.path.join(plot_dir, "pretrain_ar_metrics.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"✓ Training performance metrics plot saved to: {chart_path}")
        
        # Export metrics tracking metrics array to CSV for reference
        csv_path = os.path.join(plot_dir, "pretrain_ar_metrics.csv")
        metrics_df = pd.DataFrame({
            "step": history["steps"],
            "loss": history["train_loss"]
        })
        metrics_df.to_csv(csv_path, index=False)
        print(f"✓ Metrics exported table saved to: {csv_path}")

    except Exception as plt_err:
        print(f"⚠ Plotting runtime execution aborted: {plt_err}")
    # ────────────────────────────────────────────────────────────────

    print("Pretraining complete!")