import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SelfAttention(nn.Module):
    """Causal multi-head self-attention using PyTorch's fused SDPA kernel."""

    def __init__(self, embed_size, heads, dropout):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size needs to be divisible by heads"
        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, L, E = x.shape
        q, k, v = (
            self.qkv(x).view(N, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )  # each: (N, heads, L, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,  # right-padding means real tokens never see PAD, so no pad mask needed
        )
        out = out.transpose(1, 2).reshape(N, L, E)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN residual blocks
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class SegawaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=384,
        num_layers=6,
        heads=6,
        forward_expansion=4,
        dropout=0.2,
        max_length=512,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size, bias=False)
        self.fc_out.weight = self.word_embedding.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 style: shrink residual-branch output projections for deep-net stability
        for name, p in self.named_parameters():
            if name.endswith("attention.fc_out.weight") or name.endswith("feed_forward.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        N, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out)
        return self.fc_out(self.norm(out))


if __name__ == "__main__":
    vocab_size = 8000
    model = SegawaModel(vocab_size=vocab_size, max_length=60).to(device)
    x = torch.randint(0, vocab_size, (2, 60)).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Causality test: changing a future token must not change earlier outputs
    model.eval()
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % vocab_size
    with torch.no_grad():
        diff = (model(x)[:, :-1] - model(x2)[:, :-1]).abs().max().item()
    print(f"Causality check (should be ~0): {diff:.2e}")