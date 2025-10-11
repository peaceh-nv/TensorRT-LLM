import ctypes
from collections.abc import Sequence
from functools import lru_cache
from typing import Iterator

import cuda.bindings.driver as drv
from cuda.core.experimental import Kernel, Program, ProgramOptions
from kv_cache_manager_v2._common import (CudaStream, LayerId, MemAddress,
                                         TokenIdExt)
from kv_cache_manager_v2._utils import _unwrap, chunked, div_up, exact_div

MAX_TOKENS = 32


@lru_cache(maxsize=None)
def get_program(debug: bool = False):
    code = r"""
#if !defined(__CUDACC_RTC__)
#include <cassert>
#include <cstdio>
#endif

#ifdef NDEBUG
__device__ inline void check(bool condition) {
    if (!condition) {
        asm volatile("trap;" ::: "memory");
    }
}
#else
#define check assert
#endif

using uint32_t = unsigned int;
using uint16_t = unsigned short;

struct alignas(16) Value {
    uint32_t token;
    uint32_t layer;
    uint32_t role;
    uint32_t beam;

    __device__ inline bool operator==(Value const& other) const {
        return token == other.token && layer == other.layer && role == other.role && beam == other.beam;
    }
    __device__ inline bool operator!=(Value const& other) const {
        return !(*this == other);
    }
};

constexpr uint32_t kMAX_TOKENS = _MAX_TOKENS_;

struct Tokens {
    uint32_t tokens[kMAX_TOKENS];
};

extern "C" __global__ void fillValues(Value* data, uint32_t valuesPerToken, uint32_t layer, uint32_t buf_id, uint32_t beam, __grid_constant__ const Tokens tokens) {
    auto const tidx = (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const stride = static_cast<uint32_t>(blockDim.x) * gridDim.x;
    auto const numTokens = gridDim.y;
    auto const idxToken = blockIdx.y;
    check(numTokens <= kMAX_TOKENS);
    auto const base = data + idxToken * valuesPerToken;
    auto const token = tokens.tokens[idxToken];
    auto const value = Value{token, layer, buf_id, beam};
    for (auto idx = tidx; idx < valuesPerToken; idx += stride) {
        base[idx] = value;
    }
}

__device__ inline void assertEq(Value const& a, Value const& b) {
#ifndef NDEBUG
    if (a != b) {
        printf("(%d, %d, %d, %d) != (%d, %d, %d, %d)\n",
                a.token, a.layer, a.role, a.beam,
                b.token, b.layer, b.role, b.beam);
    }
#endif
    check(a == b);
}

extern "C" __global__ void checkValues(Value const* data, uint32_t valuesPerToken, uint32_t layer, uint32_t buf_id, uint32_t beam, __grid_constant__ const Tokens tokens) {
    auto const tidx = (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const stride = static_cast<uint32_t>(blockDim.x) * gridDim.x;
    auto const numTokens = gridDim.y;
    auto const idxToken = blockIdx.y;
    check(numTokens <= kMAX_TOKENS);
    auto const base = data + idxToken * valuesPerToken;
    auto const token = tokens.tokens[idxToken];
    auto const ref = Value{token, layer, buf_id, beam};
    for (auto idx = tidx; idx < valuesPerToken; idx += stride) {
        assertEq(base[idx], ref);
    }
}
    """
    program_options = ProgramOptions(std="c++17",
                                     lineinfo=True,
                                     debug=debug,
                                     define_macro=[("_MAX_TOKENS_",
                                                    str(MAX_TOKENS))])
    if not debug:
        program_options.use_fast_math = True
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("fillValues", "checkValues"))
    return mod


@lru_cache(maxsize=None)
def get_kernel(name: str) -> Kernel:
    assert name in ("fillValues", "checkValues")
    debug = False
    # debug = not NDEBUG
    return get_program(debug).get_kernel(name)


class Value(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_uint32),
        ("layer", ctypes.c_uint32),
        ("buf_id", ctypes.c_uint32),
        ("beam", ctypes.c_uint32),
    ]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self.token == other.token and self.layer == other.layer and self.buf_id == other.buf_id and self.beam == other.beam

    def __str__(self) -> str:
        return f"Value(token={self.token}, layer={self.layer}, buf_id={self.buf_id}, beam={self.beam})"


class Tokens(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_uint32 * MAX_TOKENS),
    ]


def _make_tokens(tokens: Sequence[TokenIdExt]) -> Tokens:
    assert len(tokens) <= MAX_TOKENS
    padded = list(tokens) + [0] * (MAX_TOKENS - len(tokens))
    return Tokens(tokens=(ctypes.c_uint32 * MAX_TOKENS)(*[
        t if isinstance(t, int) else int.
        from_bytes(t[:4], 'little', signed=False) for t in padded
    ]))


def fill_values(address: MemAddress, bytes_per_token: int, layer: LayerId,
                buf_id: int, beam: int, tokens: Sequence[TokenIdExt],
                stream: CudaStream):
    values_per_token = exact_div(bytes_per_token, ctypes.sizeof(Value))
    kernel = get_kernel("fillValues")
    for chunk in chunked(tokens, MAX_TOKENS):
        args = (address, values_per_token, layer, buf_id, beam,
                _make_tokens(chunk))
        arg_types = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                     ctypes.c_uint32, ctypes.c_uint32, None)
        _unwrap(
            drv.cuLaunchKernel(kernel._handle, div_up(values_per_token, 1024),
                               len(chunk), 1, 256, 1, 1, 0, stream,
                               (args, arg_types), 0))


def check_values(address: MemAddress, bytes_per_token: int, layer: LayerId,
                 buf_id: int, beam: int, tokens: Sequence[TokenIdExt],
                 stream: CudaStream):
    values_per_token = exact_div(bytes_per_token, ctypes.sizeof(Value))
    kernel = get_kernel("checkValues")
    for chunk in chunked(tokens, MAX_TOKENS):
        args = (address, values_per_token, layer, buf_id, beam,
                _make_tokens(chunk))
        arg_types = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                     ctypes.c_uint32, ctypes.c_uint32, None)
        _unwrap(
            drv.cuLaunchKernel(kernel._handle, div_up(values_per_token, 1024),
                               len(chunk), 1, 256, 1, 1, 0, stream,
                               (args, arg_types), 0))


def debug_dump_tokens(addr: MemAddress, token_bytes: int, num_tokens: int,
                      stream: CudaStream) -> Iterator[Value]:
    if (num_tokens == 0):
        return
    val_size = ctypes.sizeof(Value)
    values_per_token = exact_div(token_bytes, val_size)
    host_buf = (Value * values_per_token * num_tokens)()
    ptr = ctypes.addressof(host_buf)
    _unwrap(drv.cuMemcpyDtoHAsync(ptr, addr, num_tokens * token_bytes, stream))
    _unwrap(drv.cuStreamSynchronize(stream))
    for i in range(num_tokens):
        token = host_buf[i]
        value = Value.from_buffer_copy(token[0])
        for j in range(1, values_per_token):
            assert token[j] == token[0]
        yield value
