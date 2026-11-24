from my_attention import MyAttention
from timm.models import create_model
from my_attention_real_pruning import MyAttentionRealPruning

import torch
import sys
import parameters as params

from torch.nn.attention import SDPBackend, sdpa_kernel



from bench_model import benchmark_model, compute_flops, detect_sdpa_backend


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    
    img_size = 1024
    B = 1
    x = torch.randn((B,3,img_size,img_size), device=device)
    
    model_type = 'deit_small_patch16_224'
    
    
    # no bias, no topk
    original = create_model(
        model_type,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=img_size,
        # attn_layer=MyAttentionRealPruning
    )
    
        
    # bias, topk
    ours = create_model(
        model_type,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=img_size,
        attn_layer=MyAttentionRealPruning
    )
    
    for i, blk in enumerate(ours.blocks):
        blk.attn.b = torch.nn.Linear(blk.attn.attn_dim, blk.attn.num_heads, bias=False)
        torch.nn.init.zeros_(blk.attn.b.weight)
            
        blk.attn.bias_topk = params.bias_topk[i]
        
    
    # bias, no topk
    ours_no_topk = create_model(
        model_type,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=img_size,
        attn_layer=MyAttentionRealPruning
    )
    
    for i, blk in enumerate(ours_no_topk.blocks):
        blk.attn.b = torch.nn.Linear(blk.attn.attn_dim, blk.attn.num_heads, bias=False)
        torch.nn.init.zeros_(blk.attn.b.weight)
            
        blk.attn.bias_topk = None 
        
        
    
    
    
    ##### Profiling #####
    
    
    
    original.eval()
    original.to(device)
    ours.eval()
    ours.to(device)
    ours_no_topk.eval()
    ours_no_topk.to(device)
    
    original_ms, original_std_ms, _ = benchmark_model(original, x, warmup=50, iters=200) #, backend=SDPBackend.MATH)
    ours_no_topk_ms, ours_no_topk_std_ms, _ = benchmark_model(ours_no_topk, x, warmup=50, iters=200) #, backend=SDPBackend.MATH)
    ours_ms, ours_std_ms, _ = benchmark_model(ours, x, warmup=50, iters=200) #, backend=SDPBackend.MATH)

    print("\n--------- RUNTIME")
    print(f"original avg ms: {original_ms:.4f} ± {original_std_ms:.4f}")
    print(f"ours no topk avg ms: {ours_no_topk_ms:.4f} ± {ours_no_topk_std_ms:.4f}")
    print(f"ours pruning avg ms: {ours_ms:.4f} ± {ours_std_ms:.4f}")
    print(f"speedup: {original_ms / ours_ms:.3f}x")


        

    for name, m in [("original", original),
                ("ours_no_topk", ours_no_topk),
                ("ours", ours)]:
        print(f"{name:14s} backend: {detect_sdpa_backend(m, x)}  "
            f"flops: {compute_flops(m, x) / 1e9:.1f} GFLOPs")
