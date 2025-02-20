import torch
import triton
import triton.language as tl

# @triton.autotune(configs=[
#     triton.Config({'BLOCK_H': 1}, num_warps=4),
#     triton.Config({'BLOCK_H': 1}, num_warps=8),
#     triton.Config({'BLOCK_H': 2}, num_warps=4),
#     triton.Config({'BLOCK_H': 2}, num_warps=8),
#     triton.Config({'BLOCK_H': 4}, num_warps=4),
#     triton.Config({'BLOCK_H': 4}, num_warps=8),
#     triton.Config({'BLOCK_H': 8}, num_warps=4),
#     triton.Config({'BLOCK_H': 8}, num_warps=8),
#     triton.Config({'BLOCK_H': 16}, num_warps=4),
#     triton.Config({'BLOCK_H': 16}, num_warps=8),
# ], key=['batch_size', 'head_num'])
@triton.jit
def _deepseek_rope_kernel(
    Q,
    K,
    Pos,
    Out_q,
    Out_k,
    Cos_sin_cache,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_oq_b,
    stride_oq_h,
    stride_ok_b,
    batch_size: tl.constexpr,
    head_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_block_head_id = tl.program_id(1)

    pos = tl.load(Pos + cur_batch)
    cos_ptr = Cos_sin_cache + pos * ROTARY_DIM + tl.arange(0, ROTARY_DIM // 2)
    sin_ptr = Cos_sin_cache + pos * ROTARY_DIM + ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)

    if cur_block_head_id == 0:
        offs_ok = cur_batch * stride_ok_b + tl.arange(0, ROTARY_DIM)
        offs_k_0 = cur_batch * stride_k_b + tl.arange(0, ROTARY_DIM // 2) * 2
        offs_k_1 = cur_batch * stride_k_b + tl.arange(0, ROTARY_DIM // 2) * 2 + 1

        k_0 = tl.load(K + offs_k_0)
        k_1 = tl.load(K + offs_k_1)
        o_k_0 = k_0 * cos - k_1 * sin
        o_k_1 = k_1 * cos + k_0 * sin
        o_k = tl.interleave(o_k_0, o_k_1)
        tl.store(Out_k + offs_ok, o_k)

    for block_head_start in range(BLOCK_H):
        cur_head_id = cur_block_head_id * BLOCK_H + block_head_start

        offs_oq = cur_batch * stride_oq_b + cur_head_id * stride_oq_h + tl.arange(0, ROTARY_DIM)
        offs_q_0 = cur_batch * stride_q_b + cur_head_id * stride_q_h + tl.arange(0, ROTARY_DIM // 2) * 2
        offs_q_1 = cur_batch * stride_q_b + cur_head_id * stride_q_h + tl.arange(0, ROTARY_DIM // 2) * 2 + 1
        q_0 = tl.load(Q + offs_q_0)
        q_1 = tl.load(Q + offs_q_1)
        o_q_0 = q_0 * cos - q_1 * sin
        o_q_1 = q_1 * cos + q_0 * sin
        o_q = tl.interleave(o_q_0, o_q_1)

        tl.store(Out_q + offs_oq, o_q)


def deepseek_rope(positions, q, k, cos_sin_cache, rotary_dim, inplace=False):
    assert q.shape[-1] == rotary_dim
    assert k.shape[-1] == rotary_dim
    assert q.dim() == 3
    assert k.dim() == 3
    assert cos_sin_cache.shape[-1] == rotary_dim

    bs, head_num = q.shape[0], q.shape[1]
    assert k.shape[1] == 1

    if inplace:
        out_q = q
        out_k = k
    else:
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

    BLOCK_H = min(triton.cdiv(triton.next_power_of_2(bs), 128), head_num)
    # grid = lambda meta: (bs, triton.cdiv(head_num, meta['BLOCK_H']), 1)
    grid = lambda meta: (bs, triton.cdiv(head_num, BLOCK_H), 1)
    _deepseek_rope_kernel[grid](
        q,
        k,
        positions,
        out_q,
        out_k,
        cos_sin_cache,
        # stride
        q.stride(0),
        q.stride(1),
        k.stride(0),
        out_q.stride(0),
        out_q.stride(1),
        out_k.stride(0),
        batch_size=bs,
        head_num=head_num,
        BLOCK_H=BLOCK_H,
        ROTARY_DIM=rotary_dim,
    )
    return out_q, out_k