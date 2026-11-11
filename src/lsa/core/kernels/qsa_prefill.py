"""QSA serving math: block keys and index queries from the attention input, relu-scored selection.

The QSA indexer (Qwen's own sparse attention, `QSAConfig`) is an MQA indexer on the attention
input `x`: `query_head_count` query heads and one key head of `head_dim`, keys mean-pooled into
blocks of `block_size` before the rotation, which turns the first `rotary_dim` dims at the model's
frequencies - a query at its own position, a block at its first. A block's score is
`sum_h relu(q_h . k_b) / sqrt(head_dim)` over the blocks the query has closed, and the read is the
`top_k / block_size` best closed blocks expanded to keys plus the query's own open block.

Weight layout is the distilled flat dict (`lsa-qsa *.pt`, per layer): `index_qk_proj`
[(H + 1) head_dim, model_dim] with the query heads' rows first, `q_layernorm` / `k_layernorm`
(head_dim Gemma gains), `inverse_frequencies` [rotary_dim / 2] and `attention_scaling`.
"""

import logging

import torch

_LOG = logging.getLogger(__name__)


def gemma_norm(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + gain.float())


def rotate_at(x: torch.Tensor, positions: torch.Tensor, inverse_frequencies: torch.Tensor, scaling: float) -> torch.Tensor:
    half = inverse_frequencies.shape[0]
    ang = positions.float()[:, None] * inverse_frequencies[None]
    cos = (ang.cos() * scaling).view(x.shape[0], *([1] * (x.ndim - 2)), half)
    sin = (ang.sin() * scaling).view(x.shape[0], *([1] * (x.ndim - 2)), half)
    a, b, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, rest), dim=-1)


class QsaLayerIndexer:
    """One layer's indexer on NPU: projections, norms and frequencies held once."""

    def __init__(self, weights: dict, device, eps: float = 1e-6):
        head_count = weights["query_head_count"]
        head_dim = weights["head_dim"]
        qk = weights["index_qk_proj"]
        self.wq = qk[: head_count * head_dim].to(device=device, dtype=torch.bfloat16).t().contiguous()
        self.wk = qk[head_count * head_dim :].to(device=device, dtype=torch.bfloat16).t().contiguous()
        self.q_gain = weights["q_layernorm"].to(device=device, dtype=torch.float32)
        self.k_gain = weights["k_layernorm"].to(device=device, dtype=torch.float32)
        self.inv_freq = weights["inverse_frequencies"].to(device=device, dtype=torch.float32)
        self.scaling = float(weights["attention_scaling"])
        self.head_count = head_count
        self.head_dim = head_dim
        self.block_size = weights["block_size"]
        self.eps = eps

    def block_keys(self, block_x: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        # block_x [B, model_dim] bf16 means; keys [B, head_dim] fp32, roped at each block's first position
        keys = block_x.to(torch.bfloat16) @ self.wk
        return rotate_at(gemma_norm(keys, self.k_gain, self.eps), starts, self.inv_freq, self.scaling)

    def index_queries(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x [R, model_dim]; queries [R, H, head_dim] fp32, roped at each row's position
        queries = (x.to(torch.bfloat16) @ self.wq).view(x.shape[0], self.head_count, self.head_dim)
        return rotate_at(gemma_norm(queries, self.q_gain, self.eps), positions, self.inv_freq, self.scaling)


def block_means(x: torch.Tensor, block_size: int, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean `x` over every whole block of a row range whose first position is `start_pos`.

    Args:
        x: Rows [N, model_dim], covering positions start_pos .. start_pos + N - 1.
        block_size: Rows pooled into one block.
        start_pos: Absolute position of x's first row.

    Returns:
        Means [B, model_dim] fp32 and their first positions [B] int64. The trailing partial
        block's rows are left to the caller's carry.
    """
    n = x.shape[0]
    first = (start_pos + block_size - 1) // block_size * block_size - start_pos
    whole = (n - first) // block_size
    if whole <= 0:
        return x[:0].float(), torch.zeros(0, dtype=torch.int64, device=x.device)
    body = x[first : first + whole * block_size]
    means = body.float().view(whole, block_size, -1).mean(1)
    starts = torch.arange(first + start_pos, first + start_pos + whole * block_size, block_size, device=x.device)
    return means, starts


def closed_block_scores(
    queries: torch.Tensor, block_keys: torch.Tensor, positions: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Score every query against every block, `sum_h relu(q_h . k_b) / sqrt(head_dim)`.

    Args:
        queries: [R, H, head_dim] fp32, roped.
        block_keys: [B, head_dim] fp32, roped at block starts; block b starts at b * block_size.
        positions: Query rows' absolute positions [R] int64.
        block_size: Keys per block.

    Returns:
        Scores [R, B] fp32; blocks not fully closed by the query hold -inf.
    """
    scores = torch.einsum("rhd,bd->rhb", queries, block_keys).relu().sum(1)
    scores = scores / (block_keys.shape[1] ** 0.5)
    closed = (torch.arange(block_keys.shape[0], device=scores.device) + 1) * block_size <= (positions + 1)[:, None]
    return scores.masked_fill(~closed, float("-inf"))


def select_blocks(scores: torch.Tensor, top_blocks: int) -> torch.Tensor:
    """Each row's best `top_blocks` closed blocks, -inf rows padded with -1.

    Rows whose position closes fewer blocks than the budget pick what is closed (short rows are
    served dense by the caller anyway).
    """
    top = min(top_blocks, scores.shape[1])
    values, ids = torch.topk(scores, top, dim=1)
    return torch.where(values > float("-inf"), ids, torch.full_like(ids, -1))
