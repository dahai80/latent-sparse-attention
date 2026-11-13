import ctypes as ct
import logging
import os

import torch

log = logging.getLogger(__name__)

DT_FLOAT = 0
DT_INT32 = 3
DT_UINT8 = 4
DT_BF16 = 27

_opapi = None
_libacl = None


def _libs():
    # ctypes handles to the already-loaded ACL libs; safe to call repeatedly
    global _opapi, _libacl
    if _opapi is None:
        path = os.environ.get(
            "E44_COMPACT_OPAPI",
            "/usr/local/Ascend/cann-9.2.1/opp/vendors/customize/op_api/lib/libcust_opapi.so",
        )
        _opapi = ct.CDLL(path)
        _opapi.aclnnCompactReadGetWorkspaceSize.restype = ct.c_int
        _opapi.aclnnCompactRead.restype = ct.c_int
        _libacl = ct.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so")
        _libacl.aclCreateTensor.restype = ct.c_void_p
        log.info("compact_read aclnn libs loaded: %s", path)
    return _opapi, _libacl


def _mk_tensor(libacl, t, dtype):
    # aclTensor over a contiguous torch NPU tensor's storage
    if not t.is_contiguous():
        raise ValueError("compact_read inputs must be contiguous")
    dims = (ct.c_int64 * t.dim())(*t.shape)
    strides = (ct.c_int64 * t.dim())(*t.stride())
    ptr = ct.c_void_p(t.data_ptr())
    handle = libacl.aclCreateTensor(
        dims, t.dim(), dtype, strides, 0, 2, dims, t.dim(), ptr
    )
    if not handle:
        raise RuntimeError("aclCreateTensor failed")
    return ct.c_void_p(handle)


def _npu_stream():
    s = torch.npu.current_stream()
    return ct.c_void_p(s.npu_stream)


def compact_read(q, ck, cv, idx, valid, scale_t, out):
    # q [sb, H, D] fp32; ck/cv [U, Hkv, D] fp32; idx [sb, W] int32;
    # valid [sb, W] uint8; scale_t [1] fp32; out [sb, H, D] bf16
    opapi, libacl = _libs()
    tq = _mk_tensor(libacl, q, DT_FLOAT)
    tck = _mk_tensor(libacl, ck, DT_FLOAT)
    tcv = _mk_tensor(libacl, cv, DT_FLOAT)
    tidx = _mk_tensor(libacl, idx, DT_INT32)
    tvalid = _mk_tensor(libacl, valid, DT_UINT8)
    tscale = _mk_tensor(libacl, scale_t, DT_FLOAT)
    tout = _mk_tensor(libacl, out, DT_BF16)

    ws_size = ct.c_uint64(0)
    executor = ct.c_void_p()
    r = opapi.aclnnCompactReadGetWorkspaceSize(
        tq, tck, tcv, tidx, tvalid, tscale, tout,
        ct.byref(ws_size), ct.byref(executor),
    )
    if r != 0 or not executor.value:
        raise RuntimeError(f"aclnnCompactReadGetWorkspaceSize ret={r}")

    ws = ct.c_void_p()
    if ws_size.value:
        ws = ct.c_void_p(torch.empty(ws_size.value, dtype=torch.uint8, device="npu").data_ptr())

    stream = _npu_stream()
    r = opapi.aclnnCompactRead(ws, ws_size.value, executor, stream)
    if r != 0:
        raise RuntimeError(f"aclnnCompactRead ret={r}")
    return out


def compact_read_bf16(q_bf16, ck_bf16, cv_bf16, idx, valid, scale, out):
    # bf16 serving entry: bf16->f32 via .float() (exact widening, bf16 is truncated f32;
    # verified bit-equal to manual <<16 widening, 5000x faster on NPU). scale: python float
    sb, h, d = q_bf16.shape
    q32 = q_bf16.float()
    ck32 = ck_bf16.float()
    cv32 = cv_bf16.float()
    scale_t = torch.tensor([scale], dtype=torch.float32, device=q_bf16.device)
    valid_u8 = valid.to(torch.uint8)
    idx32 = idx.to(torch.int32)
    if out is None:
        out = torch.empty(sb, h, d, dtype=torch.bfloat16, device=q_bf16.device)
    compact_read(q32, ck32, cv32, idx32, valid_u8, scale_t, out)
    return out
