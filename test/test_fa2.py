#!/usr/bin/env python3
import argparse, math, time, os
import torch
import torch.nn.functional as F

# ---- FA2 imports (installed via flash-attn) ----
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
)
from flash_attn.bert_padding import pad_input, unpad_input


def cuda_cap_ok():
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    major = torch.cuda.get_device_capability()[0]
    # FlashAttention-2 is really tuned for Ampere+ (A100/H100/4090 etc.)  [oai_citation:4‡arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)
    if major < 8:
        return (
            False,
            f"GPU sm_{major}x is below recommended sm80+ for best FlashAttention-2 support",
        )
    return True, "OK"


def ref_sdp(q, k, v, attn_mask=None, dropout_p=0.0, causal=False):
    # Force math kernel, i.e. vanilla attention, to use as accuracy reference.
    # PyTorch's scaled_dot_product_attention expects q,k,v as (B,H,N,D) and a mask
    # where True means 'mask out / don't attend'.  [oai_citation:5‡Hugging Face](https://huggingface.co/OpenGVLab/InternViT-300M-448px/blame/d151fbf4b4634ac772baa719078e22275ae9e515/flash_attention.py?utm_source=chatgpt.com)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
        )


def max_abs_rel(a, b, eps=1e-6):
    diff = (a - b).abs()
    abs_err = diff.max().item()
    rel = diff / (b.abs() + eps)
    rel_err = rel.max().item()
    return abs_err, rel_err


@torch.inference_mode(False)
def check_forward_backward_padded(
    B, H, N, D, dtype, causal, dropout_p, device
):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    # --- Forward (reference PyTorch SDP) ---
    out_ref = ref_sdp(
        q, k, v, attn_mask=None, dropout_p=dropout_p, causal=causal
    )
    loss_ref = out_ref.float().sum()
    loss_ref.backward()
    gq_ref = q.grad.detach().clone()
    gk_ref = k.grad.detach().clone()
    gv_ref = v.grad.detach().clone()

    # Reset grads for FA2 pass
    q.grad = k.grad = v.grad = None

    # --- Forward (FlashAttention-2, padded API) ---
    out_fa2 = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=None,  # let kernel pick 1/sqrt(D)
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    )
    loss_fa2 = out_fa2.float().sum()
    loss_fa2.backward()
    gq_fa2 = q.grad
    gk_fa2 = k.grad
    gv_fa2 = v.grad

    # Compare outputs and grads
    oa, orl = max_abs_rel(out_ref.detach(), out_fa2.detach(), eps=1e-6)
    gqa, gql = max_abs_rel(gq_ref, gq_fa2, eps=1e-6)
    gka, gkl = max_abs_rel(gk_ref, gk_fa2, eps=1e-6)
    gva, gvl = max_abs_rel(gv_ref, gv_fa2, eps=1e-6)

    return dict(
        out_abs=oa,
        out_rel=orl,
        gq_abs=gqa,
        gq_rel=gql,
        gk_abs=gka,
        gk_rel=gkl,
        gv_abs=gva,
        gv_rel=gvl,
    )


@torch.inference_mode(False)
def check_forward_backward_varlen(
    B, H, N, D, dtype, causal, dropout_p, device
):
    # We test FA2's variable-length (unpadded) path by:
    # 1. generating random per-sequence lengths
    # 2. building a (B,N) key_padding_mask with True for *valid* tokens
    # 3. unpadding via flash_attn.bert_padding.unpad_input
    # 4. running flash_attn_varlen_qkvpacked_func
    # 5. padding back and comparing to PyTorch reference attention
    #
    # This mirrors how FA2 is actually wired in real MHA modules.  [oai_citation:6‡Hugging Face](https://huggingface.co/OpenGVLab/InternViT-300M-448px/blame/d151fbf4b4634ac772baa719078e22275ae9e515/flash_attention.py?utm_source=chatgpt.com)

    torch.manual_seed(1)

    # Random per-sample valid lengths, each in [N//2, N], guarantee first is full N.
    lens = torch.randint(
        low=max(1, N // 2),
        high=N + 1,
        size=(B,),
        device=device,
    )
    lens[0] = N

    # qkv: (B, N, 3, H, D)
    qkv = torch.randn(
        B, N, 3, H, D, device=device, dtype=dtype
    )

    # ------------------------------------------------------------------
    # Build key_padding_mask for unpad_input:
    # key_padding_mask[b, t] == True  => this token is REAL (keep it)
    # False => pad.
    # Shape: (B, N), dtype=bool.  [oai_citation:7‡Hugging Face](https://huggingface.co/togethercomputer/m2-bert-80M-2k/blob/main/bert_padding.py?utm_source=chatgpt.com)
    # ------------------------------------------------------------------
    key_padding_mask = torch.zeros(
        B, N, device=device, dtype=torch.bool
    )
    for b in range(B):
        L = lens[b].item()
        key_padding_mask[b, :L] = True  # mark valid tokens

    # ------------------------------------------------------------------
    # Build attn_mask for reference PyTorch SDP:
    # PyTorch scaled_dot_product_attention takes a boolean mask where
    # True means "do NOT attend here".
    #
    # We want to forbid:
    #  - queries that are padding
    #  - keys that are padding
    #
    # valid_mask: (B, N) True where token exists
    # broadcast to (B,1,N,N): allow attend only if both query & key valid.
    # So final attn_mask[b,0,i,j] = True if (i or j) is padding.
    # ------------------------------------------------------------------
    valid_mask = key_padding_mask  # (B, N) bool
    attn_mask = ~(valid_mask[:, None, :, None] & valid_mask[:, None, None, :])
    # attn_mask shape: (B,1,N,N) bool. True = block. Compatible with SDP.  [oai_citation:8‡Hugging Face](https://huggingface.co/OpenGVLab/InternViT-300M-448px/blame/d151fbf4b4634ac772baa719078e22275ae9e515/flash_attention.py?utm_source=chatgpt.com)

    # ------------------------------------------------------------------
    # Reference path (PyTorch SDP math kernel)
    # SDP expects q,k,v: (B,H,N,D). Our qkv right now is (B,N,3,H,D).
    # Transpose to (B,H,N,D) and requires_grad so we can compare grads.
    # ------------------------------------------------------------------
    q_ref = (
        qkv[:, :, 0]
        .transpose(1, 2)
        .contiguous()
        .requires_grad_(True)
    )  # (B,H,N,D)
    k_ref = (
        qkv[:, :, 1]
        .transpose(1, 2)
        .contiguous()
        .requires_grad_(True)
    )  # (B,H,N,D)
    v_ref = (
        qkv[:, :, 2]
        .transpose(1, 2)
        .contiguous()
        .requires_grad_(True)
    )  # (B,H,N,D)

    out_ref = ref_sdp(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        causal=causal,
    )  # (B,H,N,D)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()
    gq_ref = q_ref.grad.detach().clone()
    gk_ref = k_ref.grad.detach().clone()
    gv_ref = v_ref.grad.detach().clone()

    # clear grads so we can reuse qkv info
    q_ref.grad = k_ref.grad = v_ref.grad = None

    # ------------------------------------------------------------------
    # FlashAttention-2 varlen path
    #
    # Step 1: flatten qkv to (B,N,3*H*D)
    # Step 2: unpad_input(..., key_padding_mask) gives us:
    #    x_unpad: (nnz, 3*H*D)
    #    indices: (nnz,)
    #    cu_seqlens: (B+1,), int32 prefix sums
    #    max_seqlen: int
    #
    # Step 3: reshape x_unpad -> (nnz, 3, H, D) and requires_grad_
    # Step 4: call flash_attn_varlen_qkvpacked_func
    # Step 5: pad_input back -> (B,N,H,D)
    # ------------------------------------------------------------------
    B_, N_, three_, H_, D_ = qkv.shape
    assert B_ == B and N_ == N and three_ == 3 and H_ == H and D_ == D

    x = qkv.reshape(B, N, 3 * H * D)  # (B,N,3HD)

    # unpad_input expects mask with True for valid tokens,
    # which is exactly key_padding_mask.  [oai_citation:9‡Hugging Face](https://huggingface.co/togethercomputer/m2-bert-80M-2k/blob/main/bert_padding.py?utm_source=chatgpt.com)
    x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(
        x, key_padding_mask
    )
    # x_unpad: (nnz, 3*H*D)
    qkv_packed = (
        x_unpad.view(-1, 3, H, D)
        .requires_grad_(True)
    )  # (nnz,3,H,D)

    out_fa2 = flash_attn_varlen_qkvpacked_func(
        qkv_packed,
        cu_seqlens,
        max_seqlen,  # some installs call this arg max_seqlen_q
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=None,
    )  # (nnz,H,D)

    # Pad FA2 output back to dense:
    out_fa2_full = pad_input(
        out_fa2.reshape(-1, H * D),  # (nnz, H*D)
        indices,
        B,
        N,
    )
    
    out_fa2_full = out_fa2_full.view(B, N, H, D)  # (B,N,H,D)

    # Transpose to match out_ref layout (B,H,N,D)
    out_fa2_full_t = out_fa2_full.transpose(1, 2).contiguous()

    loss_fa2 = out_fa2_full_t.float().sum()
    loss_fa2.backward()

    # Recover grads on q,k,v from qkv_packed.grad:
    g_packed = qkv_packed.grad  # (nnz,3,H,D)
    g_full = pad_input(
        g_packed.reshape(-1, 3 * H * D),  # (nnz,3HD)
        indices,
        B,
        N,
    )  # (B,N,3HD)
    g_full = g_full.view(B, N, 3, H, D)  # (B,N,3,H,D)

    # Slice grads for q,k,v, transpose to (B,H,N,D) to match *_ref
    gq_fa2 = g_full[:, :, 0].transpose(1, 2).contiguous()
    gk_fa2 = g_full[:, :, 1].transpose(1, 2).contiguous()
    gv_fa2 = g_full[:, :, 2].transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Compare outputs + grads
    # ------------------------------------------------------------------
    oa, orl = max_abs_rel(
        out_ref.detach(), out_fa2_full_t.detach(), eps=1e-6
    )
    gqa, gql = max_abs_rel(gq_ref, gq_fa2, eps=1e-6)
    gka, gkl = max_abs_rel(gk_ref, gk_fa2, eps=1e-6)
    gva, gvl = max_abs_rel(gv_ref, gv_fa2, eps=1e-6)

    return dict(
        out_abs=oa,
        out_rel=orl,
        gq_abs=gqa,
        gq_rel=gql,
        gk_abs=gka,
        gk_rel=gkl,
        gv_abs=gva,
        gv_rel=gvl,
    )


def benchmark(
    B, H, N, D, dtype, causal, dropout_p, iters=50, warmup=10, device="cuda"
):
    torch.manual_seed(123)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(q)

    def timeit(fn_forward, name):
        # warmup
        for _ in range(warmup):
            out = fn_forward()
            (out * 0).sum().backward()
            q.grad = k.grad = v.grad = None
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn_forward()
            out.backward(grad_out)
            q.grad = k.grad = v.grad = None
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        mem = torch.cuda.max_memory_allocated() / (1024**2)
        return ms, mem

    def f_ref():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=causal,
            )

    def f_fa2():
        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
        )

    ref_ms, ref_mem = timeit(f_ref, "ref")
    fa2_ms, fa2_mem = timeit(f_fa2, "fa2")
    return dict(
        ref_ms=ref_ms,
        fa2_ms=fa2_ms,
        speedup=ref_ms / fa2_ms,
        ref_mem_mb=ref_mem,
        fa2_mem_mb=fa2_mem,
        mem_saving_mb=ref_mem - fa2_mem,
    )


def parse_dtype(s):
    s = s.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main():
    p = argparse.ArgumentParser(
        description="FlashAttention-2 test / benchmark"
    )
    p.add_argument("--B", type=int, default=2, help="batch size")
    p.add_argument("--H", type=int, default=16, help="num heads")
    p.add_argument("--N", type=int, default=1024, help="seq len")
    p.add_argument("--D", type=int, default=64, help="head dim")
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    p.add_argument(
        "--causal", action="store_true", help="use causal mask"
    )
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-varlen", action="store_true")
    args = p.parse_args()

    ok, msg = cuda_cap_ok()
    if not ok:
        print(f"[WARN] {msg}")

    device = "cuda"
    dtype = parse_dtype(args.dtype)

    # Quick numerical checks (padded)
    print("=== Numerical check: padded ===")
    res = check_forward_backward_padded(
        args.B,
        args.H,
        args.N,
        args.D,
        dtype,
        args.causal,
        args.dropout,
        device,
    )
    for k, v in res.items():
        print(f"{k:>10s}: {v:.3e}")

    # Varlen numerical checks
    if not args.skip_varlen:
        print("\n=== Numerical check: varlen ===")
        res_vl = check_forward_backward_varlen(
            args.B,
            args.H,
            args.N,
            args.D,
            dtype,
            args.causal,
            args.dropout,
            device,
        )
        for k, v in res_vl.items():
            print(f"{k:>10s}: {v:.3e}")

    # Benchmark
    print("\n=== Benchmark (ms/iter & MB) ===")
    bm = benchmark(
        args.B,
        args.H,
        args.N,
        args.D,
        dtype,
        args.causal,
        args.dropout,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    print(
        f"ref:  {bm['ref_ms']:.3f} ms | {bm['ref_mem_mb']:.1f} MB"
    )
    print(
        f"fa2:  {bm['fa2_ms']:.3f} ms | {bm['fa2_mem_mb']:.1f} MB"
    )
    print(
        f"speedup x{bm['speedup']:.2f} | mem saved {bm['mem_saving_mb']:.1f} MB"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()