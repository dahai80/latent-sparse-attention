"""Sparse read executed with torch_npu._npu_paged_attention (M1).

The oracle `topk_attention` reads keep = selection U window U sinks with one causal softmax.
This module produces the same keep set per query, gathers those kv rows into a compact
private block cache, and hands the whole batch to the production paged-attention op in
chunks of 256 queries (the op's batch ceiling).

Semantics matched to the oracle:
- keep ids are absolute key positions, causal, window then sinks merged, PAD dropped;
- one softmax over the whole keep set, fp32 accumulation inside the op;
- output cast back to q's dtype.

Known op constraints (see REPORT 20.5): the first call per process fails cleanly and must
be retried; context_lens must be a device tensor; the cache is 3D [NB, BSZ, Hkv * D].
"""

import logging

import torch

PAD = -1
_LOG = logging.getLogger(__name__)
_BMM_T = [0.0, 0.0, 0.0, 0.0, 0]


def build_keep_ids(
    selected_key_ids: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    sink: int,
    key_count: int,
    max_keep: int | None = None,
) -> torch.Tensor:
    """Return sorted, deduplicated, causal keep ids per query, right-padded with PAD.

    Args:
        selected_key_ids: Absolute key positions [N, budget] int32/64, PAD for none.
        positions: Absolute position of each query row [N] int64. Rows of one batch
            share the batch's key sequence, so callers pass per-batch positions, not
            one monotonically increasing range across batches.
        window: Recent keys each query reads, its own included.
        sink: Initial keys every query reads.
        key_count: Number of available key positions.

    Returns:
        Keep ids [N, L] int64 sorted ascending, PAD-padded on the right.
    """
    device = selected_key_ids.device
    n, budget = selected_key_ids.shape
    positions = positions.to(device=device, dtype=torch.int64)

    window_ids = positions[:, None] - torch.arange(window, device=device)[None, :]
    sink_ids = torch.arange(sink, device=device)[None, :].expand(n, -1)
    sel = selected_key_ids.long()
    sel_valid = sel != PAD
    sel = torch.where(sel_valid, sel, torch.zeros_like(sel))
    cat = torch.cat([sel, window_ids, sink_ids], dim=1)
    valid = torch.cat(
        [
            sel_valid,
            torch.ones_like(window_ids, dtype=torch.bool),
            torch.ones_like(sink_ids, dtype=torch.bool),
        ],
        dim=1,
    )

    cat = cat.clamp(0, key_count - 1)
    cat = torch.where(cat > positions[:, None], torch.zeros_like(cat), cat)
    cat = torch.where(valid & (cat <= positions[:, None]), cat, torch.full_like(cat, PAD))

    sorted_ids, order = torch.sort(cat, dim=1)
    live = sorted_ids != PAD
    first = live & torch.cat(
        [torch.ones_like(sorted_ids[:, :1], dtype=torch.bool), sorted_ids[:, 1:] != sorted_ids[:, :-1]], dim=1
    )
    counts = first.sum(dim=1)
    if max_keep is None:
        max_keep = int(counts.max().item()) if n else 0

    flat_rank = torch.cumsum(first.long(), dim=1) - 1
    keep = torch.full((n, max_keep), PAD, dtype=torch.int64, device=device)
    row = torch.arange(n, device=device)[:, None].expand_as(sorted_ids)
    keep[row[first], flat_rank[first]] = sorted_ids[first]
    _LOG.debug("keep ids: n=%d budget=%d max_keep=%d mean=%.1f", n, budget, max_keep, counts.float().mean().item())
    return keep, counts


def paged_attention_pooled(
    q_flat: torch.Tensor,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    keep: torch.Tensor,
    counts: torch.Tensor,
    scale: float,
    block_size: int = 128,
    chunk: int = 256,
    backend: str = "fa",
    gather=None,
) -> torch.Tensor:
    """Attention over each query's keep set, rows gathered from a flat slot-indexed KV pool.

    Serving form of `paged_topk_attention`: keys and values live in the paged cache flattened to
    [slots, Hkv, Dh], and `keep` holds absolute positions mapped to slots by the caller's slot
    table before this call.

    Two backends:
    - "bmm": the serving backend. Per 256-row sub-batch: gather kv (callers pass a `gather` that
      reads the paged cache into persistent buffers), QK einsum, one fp32 softmax with the PAD
      mask (rows need no grouping by keep count), PV einsum. ~4 TB/s effective on the gathered
      kv vs ~43 GB/s for the fusion attention at these GQA-broadcast shapes.
    - "fa": rows are grouped by their exact keep count, each group's keep rows are gathered into a
      per-row batch [g, L, Hkv, Dh] and read with `npu_fusion_attention` (one query per batch row,
      no padding, so no mask). The mask input of the fusion attention is unreliable at large key
      widths on CANN 9.2.1 (returns wrong output for L >= ~2052, exact at L = 16), which is why
      rows are grouped instead of masked. Kept as a reference backend.
    - "v2": the production paged-attention op. **One call per process** - a second invocation
      faults the device even with identical inputs (REPORT 20.14), so this is offline/parity only.

    Args:
        q_flat: Queries [N, H, Dh].
        pool_k / pool_v: Key / value pools [S, Hkv, Dh], slot-indexed.
        keep: Keep slot ids [N, L] int64, PAD (-1) right-padded, sorted ascending per row.
        counts: Live keep entries per row [N] int64.
        scale: Multiplier applied to query-key scores.
        block_size: Compact cache block the v2 backend packs the gathered rows into.
        chunk: Queries per call, at most 256.
        backend: "fa" (serving) or "v2" (one-shot).

    Returns:
        Attention output [N, H, Dh], q's dtype and device.
    """
    if backend == "fa":
        return _pooled_fa_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather)
    if backend == "bmm":
        return _pooled_bmm_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather)
    if backend == "compact":
        return _pooled_compact_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather)
    return _pooled_v2_read(q_flat, pool_k, pool_v, keep, counts, scale, block_size, chunk)


def _pooled_bmm_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather=None):
    # manual GQA attention over each row's keep set: QK bmm, one fp32 softmax over the
    # whole keep row (PAD masked), PV bmm. npu_fusion_attention reaches only ~0.1 ms per
    # query row at these shapes (GQA broadcast, ~43 GB/s effective); the bmm form reads the
    # gathered kv twice at full bandwidth. Rows are processed in sub-batches of `chunk`;
    # rows with different keep counts need no grouping - PAD scores are masked.
    import os
    import time

    bmm_timed = os.environ.get("VLLM_BMM_TIME") == "1"
    t_gather = t_qk = t_sm = t_pv = 0.0
    if gather is None:
        gather = lambda ids: (pool_k[ids], pool_v[ids])

    n, head_count, head_dim = q_flat.shape
    kv_head_count = pool_k.shape[1] if pool_k is not None else None
    out = torch.empty(n, head_count, head_dim, device=q_flat.device, dtype=q_flat.dtype)
    neg_inf = float("-inf")
    done = 0
    while done < n:
        end = min(done + chunk, n)
        rows_keep = keep[done:end]
        sb, width = rows_keep.shape
        if bmm_timed:
            torch.npu.synchronize()
            t_a = time.perf_counter()
        gk, gv = gather(rows_keep.clamp(0))
        gk = gk.view(sb, width, -1, head_dim)
        gv = gv.view(sb, width, -1, head_dim)
        if bmm_timed:
            torch.npu.synchronize()
            t_b = time.perf_counter()
            t_gather += t_b - t_a
        kv_head_count = gk.shape[2]
        hpg = head_count // kv_head_count
        q_b = q_flat[done:end].view(sb, kv_head_count, hpg, head_dim)
        # callers may map PAD ids onto a live slot (e.g. slot_table[keep.clamp(0)]), so the
        # padding is detected from counts, not from sentinel values in rows_keep
        pad = torch.arange(width, device=q_b.device)[None, :] >= counts[done:end].to(torch.int64)[:, None]
        scores = torch.einsum("bgqd,bsgd->bgqs", q_b, gk).float() * scale
        scores.masked_fill_(pad[:, None, None, :], neg_inf)
        if bmm_timed:
            torch.npu.synchronize()
            t_c = time.perf_counter()
            t_qk += t_c - t_b
        probs = torch.softmax(scores, dim=-1).to(q_b.dtype)
        if bmm_timed:
            torch.npu.synchronize()
            t_d = time.perf_counter()
            t_sm += t_d - t_c
        o = torch.einsum("bgqs,bsgd->bgqd", probs, gv)
        out[done:end] = o.reshape(sb, head_count, head_dim)
        if bmm_timed:
            torch.npu.synchronize()
            t_e = time.perf_counter()
            t_pv += t_e - t_d
        done = end
    if bmm_timed:
        torch.npu.synchronize()
        _BMM_T[0] += t_gather
        _BMM_T[1] += t_qk
        _BMM_T[2] += t_sm
        _BMM_T[3] += t_pv
        _BMM_T[4] += 1
        if _BMM_T[4] % 64 == 0:
            print(f"BMM time: gather={_BMM_T[0]:.1f}s qk={_BMM_T[1]:.1f}s "
                  f"softmax={_BMM_T[2]:.1f}s pv={_BMM_T[3]:.1f}s calls={_BMM_T[4]}", flush=True)
    return out


def _pooled_compact_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather=None):
    # all-pairs compact attention (E41): per chunk, dedup the chunk's keep ids into the union
    # set U, gather kv only for U rows, compute QK against the union once, then map scores
    # back to per-row keep positions through the unique inverse. Mathematically identical to
    # the bmm backend when every row's keep ids are unique (build_keep_ids guarantees it);
    # duplicated ids within a row collapse to one copy here, whereas bmm counts them twice.
    import os
    import time

    bmm_timed = os.environ.get("VLLM_BMM_TIME") == "1"
    t_uniq = t_qk = t_sm = t_pv = 0.0
    if gather is None:
        gather = lambda ids: (pool_k[ids.clamp(0)], pool_v[ids.clamp(0)])

    n, head_count, head_dim = q_flat.shape
    out = torch.empty(n, head_count, head_dim, device=q_flat.device, dtype=q_flat.dtype)
    done = 0
    while done < n:
        end = min(done + chunk, n)
        rows_keep = keep[done:end]
        sb, width = rows_keep.shape
        if bmm_timed:
            torch.npu.synchronize()
            t_a = time.perf_counter()
        # int32 unique: same union (slot ids < 2^31), ~40% cheaper sort
        uniq, inv = torch.unique(rows_keep.to(torch.int32), return_inverse=True)
        uniq = uniq.to(torch.int64)
        u = uniq.shape[0]
        if bmm_timed:
            torch.npu.synchronize()
            t_b = time.perf_counter()
            t_uniq += t_b - t_a
        gku, gvu = gather(uniq.clamp(0).view(1, -1))
        ck = gku.view(-1, gku.shape[-2], head_dim)
        cv = gvu.view(-1, gvu.shape[-2], head_dim)
        kv_head_count = ck.shape[1]
        hpg = head_count // kv_head_count
        ck2 = ck.transpose(0, 1).contiguous()
        cv2 = cv.transpose(0, 1).contiguous()
        IDX = inv.unsqueeze(1).unsqueeze(2).expand(-1, kv_head_count, hpg, -1).contiguous()
        q_b = q_flat[done:end].view(sb, kv_head_count, hpg, head_dim)
        # scores stay bf16 (bmm native output, same rounding as the bmm backend's
        # bf16 bmm output); scale is applied in fp32 after the gather, halving the
        # scores_u materialization traffic
        scores_u = torch.bmm(
            q_b.permute(1, 0, 2, 3).reshape(kv_head_count, sb * hpg, head_dim), ck2.transpose(1, 2)
        )
        sc_row = scores_u.view(kv_head_count, sb, hpg, u).permute(1, 0, 2, 3).gather(3, IDX).float() * scale
        pad = torch.arange(width, device=q_b.device)[None, :] >= counts[done:end].to(torch.int64)[:, None]
        if pad.any():
            sc_row = sc_row.masked_fill(pad[:, None, None, :], float("-inf"))
        if bmm_timed:
            torch.npu.synchronize()
            t_c = time.perf_counter()
            t_qk += t_c - t_b
        probs = torch.softmax(sc_row, dim=-1).to(q_b.dtype)
        if bmm_timed:
            torch.npu.synchronize()
            t_d = time.perf_counter()
            t_sm += t_d - t_c
        probs_u = torch.zeros(kv_head_count, sb, hpg, u, device=q_flat.device, dtype=q_b.dtype)
        probs_u.scatter_(3, IDX.permute(1, 0, 2, 3).contiguous(), probs.permute(1, 0, 2, 3).contiguous())
        o = torch.bmm(probs_u.view(kv_head_count, sb * hpg, u), cv2)
        out[done:end] = o.view(kv_head_count, sb, hpg, head_dim).permute(1, 0, 2, 3).reshape(sb, head_count, head_dim)
        if bmm_timed:
            torch.npu.synchronize()
            t_e = time.perf_counter()
            t_pv += t_e - t_d
        done = end
    if bmm_timed:
        torch.npu.synchronize()
        _BMM_T[0] += t_uniq
        _BMM_T[1] += t_qk
        _BMM_T[2] += t_sm
        _BMM_T[3] += t_pv
        _BMM_T[4] += 1
        if _BMM_T[4] % 64 == 0:
            print(f"BMM time(compact): uniq={_BMM_T[0]:.1f}s qk={_BMM_T[1]:.1f}s "
                  f"softmax={_BMM_T[2]:.1f}s pv={_BMM_T[3]:.1f}s calls={_BMM_T[4]}", flush=True)
    return out


def _pooled_fa_read(q_flat, pool_k, pool_v, keep, counts, scale, chunk, gather=None):
    import torch_npu
    from torch_npu import npu_fusion_attention

    if gather is None:
        gather = lambda ids: (pool_k[ids], pool_v[ids])

    n, head_count, head_dim = q_flat.shape
    out = torch.empty(n, head_count, head_dim, device=q_flat.device, dtype=q_flat.dtype)
    live = counts.to(torch.int64).clamp_min(1)
    order = torch.argsort(live)
    sorted_live = live[order]
    done = 0
    while done < n:
        end = min(done + chunk, n)
        idx = order[done:end]
        group_live = sorted_live[done:end]
        # Rows sharing an exact keep count read together, width = that count, nothing padded.
        # Batch rows are padded UP to `chunk` with duplicated rows (their outputs land on
        # duplicated sel ids, same value) so the op sees a fixed (chunk, c) shape per width
        # class - unique shapes defeat the aclnn tiling cache and cost ~100ms per call.
        uniq = torch.unique(group_live).tolist()
        for c in uniq:
            sel = idx[group_live == c]
            m = sel.shape[0]
            # each sub-batch is padded to exactly `chunk` rows so the op sees one fixed
            # (chunk, c) shape; gathered kv stays under the 4 GB single-tensor limit
            for start in range(0, m, chunk):
                sub = sel[start : start + chunk]
                sb = sub.shape[0]
                pad_n = chunk - sb
                rows_keep = keep[sub][:, :c]
                sel_q = q_flat[sub]
                if pad_n:
                    rows_keep = torch.cat([rows_keep, rows_keep[-1:].expand(pad_n, -1)], 0)
                    sel_q = torch.cat([sel_q, sel_q[-1:].expand(pad_n, -1, -1)], 0)
                    sub = torch.cat([sub, sub[-1:].expand(pad_n)], 0)
                gathered_k, gathered_v = gather(rows_keep)
                q = sel_q.view(-1, 1, head_count, head_dim).contiguous()
                res = npu_fusion_attention(
                    q,
                    gathered_k,
                    gathered_v,
                    head_num=head_count,
                    input_layout="BSND",
                    scale=scale,
                    pre_tockens=int(c),
                    next_tockens=0,
                )
                out[sub] = res[0].view(-1, head_count, head_dim)
        done = end
    torch.npu.synchronize()
    return out


def _pooled_v2_read(q_flat, pool_k, pool_v, keep, counts, scale, block_size, chunk):
    import torch_npu

    n, head_count, head_dim = q_flat.shape
    kv_head_count = pool_k.shape[1]
    q = q_flat.contiguous()
    out = torch.empty_like(q)

    # Rows are processed in chunks: a single gather over all rows can exceed the 4 GB
    # single-tensor addressing limit on the NPU (errcode 95 / invalid GM address), and the
    # op itself takes at most `chunk` rows per call.
    done = 0
    while done < n:
        end = min(done + chunk, n)
        rows_keep = keep[done:end]
        max_keep = rows_keep.shape[1]
        max_blocks = (max_keep + block_size - 1) // block_size
        padded = max_blocks * block_size
        rows_n = end - done

        pad = torch.where(rows_keep == PAD, torch.zeros_like(rows_keep), rows_keep)
        gathered_k = pool_k[pad]
        gathered_v = pool_v[pad]
        live = rows_keep != PAD
        gathered_k = torch.where(live[:, :, None, None], gathered_k, torch.zeros_like(gathered_k))
        gathered_v = torch.where(live[:, :, None, None], gathered_v, torch.zeros_like(gathered_v))
        if padded > max_keep:
            tail_k = torch.zeros(
                rows_n, padded - max_keep, kv_head_count, head_dim, dtype=pool_k.dtype, device=q.device
            )
            tail_v = torch.zeros_like(tail_k)
            gathered_k = torch.cat([gathered_k, tail_k], dim=1)
            gathered_v = torch.cat([gathered_v, tail_v], dim=1)

        cache_k = (
            gathered_k.view(rows_n, max_blocks, block_size, kv_head_count, head_dim)
            .reshape(-1, block_size, kv_head_count, head_dim)
            .contiguous()
        )
        cache_v = (
            gathered_v.view(rows_n, max_blocks, block_size, kv_head_count, head_dim)
            .reshape(-1, block_size, kv_head_count, head_dim)
            .contiguous()
        )
        block_table = torch.arange(rows_n * max_blocks, device=q.device, dtype=torch.int32).view(rows_n, max_blocks)
        torch.ops.atb._npu_paged_attention_v2.out(
            q[done:end],
            cache_k,
            block_table,
            counts[done:end].tolist(),
            value_cache=cache_v,
            num_kv_heads=kv_head_count,
            num_heads=head_count,
            scale_value=scale,
            out=out[done:end],
        )
        done = end
    torch.npu.synchronize()
    return out


def paged_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_key_ids: torch.Tensor,
    window: int,
    sink: int,
    scale: float,
    query_start: int = 0,
    block_size: int = 128,
    chunk: int = 256,
    backend: str = "v2",
) -> torch.Tensor:
    """Attention over each query's keep set via the production paged-attention op.

    Args:
        q: Queries [B, Tq, Hq, Dh] (bf16 or fp16 on npu).
        k: Keys [B, Tk, Hkv, Dh], same dtype as q.
        v: Values [B, Tk, Hkv, Dh], same dtype as q.
        selected_key_ids: Long-range key positions [B, Tq, budget] int32, PAD for none.
        window: Recent keys each query reads.
        sink: Initial keys every query reads.
        scale: Multiplier applied to query-key scores.
        query_start: Absolute position of the first query row.
        block_size: Cache block size the compact cache is built on.
        chunk: Queries per op call, at most 256.

    Returns:
        Attention output [B, Tq, Hq, Dh], q's dtype and device.
    """
    import torch_npu

    batch_size, query_count, head_count, head_dim = q.shape
    key_count = k.shape[1]
    kv_head_count = k.shape[2]
    n = batch_size * query_count

    sel = selected_key_ids.reshape(n, -1)
    row_positions = query_start + torch.arange(query_count, device=q.device).repeat(batch_size)
    keep, counts = build_keep_ids(sel, row_positions, window, sink, key_count)
    max_keep = keep.shape[1]
    max_blocks = (max_keep + block_size - 1) // block_size

    b_idx = torch.arange(batch_size, device=q.device)[:, None].expand(-1, query_count).reshape(n)
    pad = torch.where(keep == PAD, torch.zeros_like(keep), keep)
    gathered_k = k[b_idx[:, None], pad]
    gathered_v = v[b_idx[:, None], pad]
    live = keep != PAD
    gathered_k = torch.where(live[:, :, None, None], gathered_k, torch.zeros_like(gathered_k))
    gathered_v = torch.where(live[:, :, None, None], gathered_v, torch.zeros_like(gathered_v))

    padded = max_blocks * block_size
    if padded > max_keep:
        tail = torch.zeros(n, padded - max_keep, kv_head_count, head_dim, dtype=k.dtype, device=q.device)
        gathered_k = torch.cat([gathered_k, tail], dim=1)
        gathered_v = torch.cat([gathered_v, tail], dim=1)

    if backend == "v2":
        cache_k = gathered_k.view(n, max_blocks, block_size, kv_head_count, head_dim).reshape(-1, block_size, kv_head_count, head_dim).contiguous()
        cache_v = gathered_v.view(n, max_blocks, block_size, kv_head_count, head_dim).reshape(-1, block_size, kv_head_count, head_dim).contiguous()
    else:
        cache_k = gathered_k.view(n, max_blocks, block_size, kv_head_count * head_dim).reshape(-1, block_size, kv_head_count * head_dim).contiguous()
        cache_v = gathered_v.view(n, max_blocks, block_size, kv_head_count * head_dim).reshape(-1, block_size, kv_head_count * head_dim).contiguous()
    block_table = torch.arange(n * max_blocks, device=q.device, dtype=torch.int32).view(n, max_blocks)

    q_flat = q.reshape(n, head_count, head_dim).contiguous()
    out = torch.empty_like(q_flat)

    if backend == "v2":
        torch.ops.atb._npu_paged_attention_v2.out(
            q_flat, cache_k, block_table, counts.tolist(),
            value_cache=cache_v, num_kv_heads=kv_head_count, num_heads=head_count,
            scale_value=scale, out=out)
        torch.npu.synchronize()
        return out.view(batch_size, query_count, head_count, head_dim).to(q.dtype)

    context_lens = counts.to(torch.int32)
    done = 0
    while done < n:
        end = min(done + chunk, n)
        call = lambda: torch_npu._npu_paged_attention(
            query=q_flat[done:end],
            key_cache=cache_k,
            value_cache=cache_v,
            num_kv_heads=kv_head_count,
            num_heads=head_count,
            scale_value=scale,
            block_table=block_table[done:end],
            context_lens=context_lens[done:end],
            out=out[done:end],
        )
        try:
            call()
        except RuntimeError as setup_failure:
            for attempt in range(2, 7):
                torch.npu.synchronize()
                try:
                    call()
                    _LOG.info("paged attention call ok on attempt %d (setup failed: %s)", attempt, setup_failure)
                    break
                except RuntimeError:
                    if attempt == 6:
                        raise
        done = end

    return out.view(batch_size, query_count, head_count, head_dim).to(q.dtype)
