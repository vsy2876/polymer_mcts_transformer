"""PSELFIES Autoregressive Language Model with AdaLN property conditioning."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Embedding for scalar conditioning."""

    def __init__(self, embed_dim: int = 128, scale: float = 16.0):
        super().__init__()
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) scalar values
        Returns:
            (B, embed_dim) Fourier features
        """
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.W  # (B, embed_dim//2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PSELFIESLanguageModel(nn.Module):
    """GPT-style causal transformer with AdaLN property conditioning."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 128,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # Property conditioning
        self.property_fourier = GaussianFourierProjection(embed_dim=128, scale=16.0)
        self.adaln_property = nn.Sequential(
            nn.Linear(128, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),  # scale and shift
        )

        # Transformer layers (standard pre-norm, NO AdaLN inside blocks)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Final norm + AdaLN applied once before output head
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Output head
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

        # NEW: Value head (Score prediction)
        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Bounds the expected reward between -1.0 and 1.0
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_adaln_params(self, property: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Get AdaLN scale and shift from property scalar."""
        device = next(self.parameters()).device
        if property is None:
            return torch.zeros(batch_size, self.d_model * 2, device=device)

        property = property.to(device)
        if property.dim() == 0:
            property = property.unsqueeze(0)
        if property.dim() == 1 and property.size(0) != batch_size:
            # Broadcast single property to batch
            property = property.expand(batch_size)

        # 1. Create a boolean mask tracking where our [NULL] sentinel (-1.0) resides
        # Shape remains strictly bounded at (B, 1) to allow broadcast matching
        is_null = (property == -1.0).unsqueeze(-1)

        fourier = self.property_fourier(property)  # (B, 128)
        adaln = self.adaln_property(fourier)  # (B, d_model*2)

        # 2. Functional out-of-place fill protects the graph from dynamic shape breaking
        # If is_null is True, it returns 0.0; otherwise it preserves adaln activations smoothly
        adaln = torch.where(is_null, 0.0, adaln)
            
        return adaln

    def forward(
        self,
        input_ids: torch.Tensor,
        property: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs
            property: (B,) scalar property values, or None for unconditional
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, L, d_model)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)  # (B, L, d_model)
        x = self.dropout(tok_emb + pos_emb)  # (B, L, d_model)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        )

        # Transformer blocks (standard pre-norm, no AdaLN inside)
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Apply AdaLN once before output head
        adaln_params = self.get_adaln_params(property, B)  # (B, d_model*2)
        scale, shift = adaln_params.chunk(2, dim=-1)  # each (B, d_model)

        x = self.final_norm(x)  # (B, L, d_model), no affine
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (B, L, d_model)

        logits = self.output_head(x)  # (B, L, vocab_size)
        values = self.value_head(x).squeeze(-1)    # (B, L) scalar predictions

        return logits, values


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer decoder layer (NO AdaLN inside)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout1(attn_out)

        # FFN with pre-norm
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x
