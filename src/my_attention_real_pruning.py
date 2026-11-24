import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type, Final
from torch.nn.attention import SDPBackend, sdpa_kernel

from my_attention import MyAttention


class MyAttentionRealPruning(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        dd = {"device": device, "dtype": dtype}
        dim_out = dim_out or dim

        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0
            head_dim = dim // num_heads

        if qk_norm or scale_norm:
            assert norm_layer is not None

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(
            dim,
            3 * self.attn_dim,
            bias=qkv_bias,
            **dd,
        )

        self.q_norm = (
            norm_layer(head_dim, **dd)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(head_dim, **dd)
            if qk_norm
            else nn.Identity()
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.norm = (
            norm_layer(self.attn_dim, **dd)
            if scale_norm
            else nn.Identity()
        )

        self.proj = nn.Linear(
            self.attn_dim,
            dim_out,
            bias=proj_bias,
            **dd,
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.b = None
        self.bias_topk = None

        self.register_buffer(
            "last_pruned_ratio",
            torch.zeros(num_heads),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        additive_mask = attn_mask

        if self.b is not None:
            bias = self.b(x).transpose(1, 2)  # B, H, T

            if self.bias_topk is not None:
                keep = int(self.bias_topk * T)
                keep = max(1, min(keep, T))

                topk_values, topk_indices = torch.topk(
                    bias,
                    k=keep,
                    dim=-1,
                    sorted=False,
                )

                gather_index = topk_indices.unsqueeze(-1).expand(
                    B, H, keep, D
                )

                k = torch.gather(k, dim=2, index=gather_index)
                v = torch.gather(v, dim=2, index=gather_index)

                additive_mask = topk_values.unsqueeze(2)

                self.last_pruned_ratio.fill_(
                    1.0 - keep / T
                )
            else:
                # additive_mask = bias.unsqueeze(2)
                # self.last_pruned_ratio.zero_()
                # logit = QK*scale + b_j, absorbed in an extra dim -> no attn_mask
                b_key   = bias.unsqueeze(-1)                          # (B, H, T, 1)
                q_const = q.new_full((B, H, T, 1), 1.0 / self.scale)  # constant = sqrt(D)

                q = torch.cat([q, q_const], dim=-1)                  # (B, H, T, D+1)
                k = torch.cat([k, b_key],  dim=-1)                   # (B, H, T, D+1)

                # flash requires head_dim to be a multiple of 8; zero-padding does not change the product
                pad = (-(self.head_dim + 1)) % 8
                if pad:
                    q = F.pad(q, (0, pad))
                    k = F.pad(k, (0, pad))

                additive_mask = None
                self.last_pruned_ratio.zero_()
        else:
            self.last_pruned_ratio.zero_()

        dropout_p = self.attn_drop.p if self.training else 0.0

        # Let PyTorch choose the backend whenever possible.
        if self.b is not None and self.bias_topk is None:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=additive_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale = self.scale
            )
        else:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=additive_mask,
                dropout_p=dropout_p,
                is_causal=is_causal
            )


        out = out.transpose(1, 2).reshape(
            B, T, self.attn_dim
        )

        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
    
    
    
    
    
def test_hard_vs_soft():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    B, N, C = 2, 384, 192

    x = torch.randn(B, N, C, device=device)

    attn_soft = MyAttention(
        dim=192,
        num_heads=3,
        attn_head_dim=64,
        dim_out=192,
        qkv_bias=True,
        qk_norm=False,
        scale_norm=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    ).to(device)
    
    attn_soft.b = nn.Linear(
        in_features=192,
        out_features=3,
        bias=False,
    ).to(device)
    attn_soft.bias_topk = 0.9

    attn_real = MyAttentionRealPruning(
        dim=192,
        num_heads=3,
        attn_head_dim=64,
        dim_out=192,
        qkv_bias=True,
        qk_norm=False,
        scale_norm=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    ).to(device)

    attn_real.b = nn.Linear(
        in_features=192,
        out_features=3,
        bias=False,
    ).to(device)
    attn_real.bias_topk = 0.9
    
    
    # Same weights
    attn_real.load_state_dict(attn_soft.state_dict())


    attn_soft.eval()
    attn_real.eval()

    with torch.no_grad():
        y_soft = attn_soft(x)
        y_real = attn_real(x)

    print("y_soft shape:", y_soft.shape)
    print("y_real shape:", y_real.shape)

    print("max abs diff output:", (y_soft - y_real).abs().max().item())
    print("mean abs diff output:", (y_soft - y_real).abs().mean().item())

    print(
        "outputs close:",
        torch.allclose(y_soft, y_real, atol=1e-5, rtol=1e-5)
    )
    
    
if __name__ == "__main__":
    test_hard_vs_soft()
