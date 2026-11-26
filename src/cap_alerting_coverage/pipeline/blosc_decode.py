"""Portable ctypes Blosc (LZ4 / Zstd / byte- & bit-shuffle) chunk decoder with bounds guards."""

from __future__ import annotations

import ctypes
import ctypes.util
import struct

import numpy as np

MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024  # 256 MiB per Zarr chunk

_liblz4 = None
_libzstd = None


def _load_native_lib(name: str, fallback_sonames: tuple[str, ...]):
    lib_path = ctypes.util.find_library(name)
    candidates = ([lib_path] if lib_path else []) + list(fallback_sonames)
    last_err = None
    for cand in candidates:
        try:
            return ctypes.CDLL(cand)
        except OSError as exc:
            last_err = exc
    raise ImportError(f"Required native compression library '{name}' could not be loaded.") from last_err


def _get_liblz4():
    global _liblz4
    if _liblz4 is None:
        lib = _load_native_lib("lz4", ("liblz4.so.1", "liblz4.so"))
        lib.LZ4_decompress_safe.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.LZ4_decompress_safe.restype = ctypes.c_int
        _liblz4 = lib
    return _liblz4


def _get_libzstd():
    global _libzstd
    if _libzstd is None:
        lib = _load_native_lib("zstd", ("libzstd.so.1", "libzstd.so"))
        lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_decompress.restype = ctypes.c_size_t
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        _libzstd = lib
    return _libzstd


def unbitshuffle_block(block_raw: bytes, typesize: int) -> bytes:
    """Invert Blosc bitshuffle transform on a decoded block."""
    if typesize <= 0:
        raise ValueError(f"Invalid Blosc typesize: {typesize}")
    bsize = len(block_raw)
    nelems = bsize // typesize
    n_bshuf = (nelems // 8) * 8
    if n_bshuf == 0:
        return block_raw
    nbytes_bshuf = n_bshuf * typesize
    nbits = typesize * 8
    bits = np.unpackbits(np.frombuffer(block_raw, dtype=np.uint8, count=nbytes_bshuf), bitorder="little")
    bits_un = bits.reshape(nbits, n_bshuf).T.reshape(-1)
    unshuf = np.packbits(bits_un, bitorder="little").tobytes()
    if bsize > nbytes_bshuf:
        unshuf += block_raw[nbytes_bshuf:]
    return unshuf


def decode_blosc(buf: bytes, max_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """Decode a Blosc v1 compressed chunk safely with strict header and stream bounds checks."""
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise TypeError("Blosc buffer must be bytes-like.")
    buf_len = len(buf)
    if buf_len < 16:
        raise ValueError(f"Truncated Blosc header: {buf_len} < 16 bytes.")

    _, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBIII", buf[:16])
    if typesize == 0:
        raise ValueError("Invalid Blosc header: typesize == 0.")
    if nbytes > max_bytes:
        raise ValueError(f"Blosc nbytes {nbytes} exceeds max_bytes {max_bytes}.")

    memcpyed = bool(flags & 0x02)
    shuffle = bool(flags & 0x01)
    bitshuffle = bool(flags & 0x04)
    compcode = (flags >> 5) & 0x07  # 1=lz4, 4=zstd

    if memcpyed:
        if 16 + nbytes > buf_len:
            raise ValueError(f"Truncated memcpyed Blosc payload: need {16 + nbytes}, have {buf_len}.")
        return bytes(buf[16 : 16 + nbytes])

    if nbytes == 0:
        return b""
    if blocksize == 0:
        raise ValueError("Invalid Blosc header: blocksize == 0.")

    nblocks = nbytes // blocksize
    leftover = nbytes % blocksize
    if leftover > 0:
        nblocks += 1

    header_end = 16 + 4 * nblocks
    if cbytes < header_end or cbytes > buf_len:
        raise ValueError(f"Invalid Blosc cbytes={cbytes} for header_end={header_end}, buf_len={buf_len}.")

    bstarts = struct.unpack(f"<{nblocks}i", buf[16:header_end])
    out = bytearray(nbytes)

    for i in range(nblocks):
        bsize = leftover if (i == nblocks - 1 and leftover > 0) else blocksize
        pos = bstarts[i]
        next_pos = bstarts[i + 1] if (i + 1 < nblocks) else cbytes
        if pos < header_end or pos + 4 > cbytes or next_pos < pos or next_pos > cbytes:
            raise ValueError(f"Invalid Blosc block bounds: pos={pos}, next_pos={next_pos}, cbytes={cbytes}.")

        first_csize = struct.unpack("<i", buf[pos : pos + 4])[0]
        if (pos + 4 + first_csize == next_pos) or (first_csize == bsize and pos + 4 + bsize == next_pos):
            nstreams = 1
            neblock = bsize
        else:
            if bsize % typesize != 0:
                raise ValueError(f"Split Blosc block size {bsize} not divisible by typesize {typesize}.")
            nstreams = typesize
            neblock = bsize // typesize
        if neblock <= 0:
            raise ValueError(f"Invalid Blosc stream size neblock={neblock}.")

        block_raw = bytearray(bsize)
        raw_offset = 0
        for _ in range(nstreams):
            if pos < header_end or pos + 4 > cbytes:
                raise ValueError(f"Truncated Blosc stream header at pos={pos}, cbytes={cbytes}.")
            csize = struct.unpack("<i", buf[pos : pos + 4])[0]
            pos += 4
            if raw_offset + neblock > bsize:
                raise ValueError(f"Blosc output overflow: {raw_offset + neblock} > {bsize}.")
            if csize < 0:
                raise ValueError(f"Invalid negative Blosc stream csize={csize}.")
            if csize == 0:
                raw_offset += neblock
                continue
            if csize > cbytes - pos:
                raise ValueError(f"Truncated Blosc stream: csize={csize} exceeds remaining {cbytes - pos} bytes.")
            if csize == neblock:
                block_raw[raw_offset : raw_offset + neblock] = buf[pos : pos + neblock]
            else:
                cbuf = bytes(buf[pos : pos + csize])
                if len(cbuf) != csize:
                    raise ValueError(f"Short Blosc stream slice: {len(cbuf)} != {csize}.")
                dbuf = ctypes.create_string_buffer(neblock)
                if compcode == 1:
                    liblz4 = _get_liblz4()
                    res = liblz4.LZ4_decompress_safe(cbuf, dbuf, len(cbuf), neblock)
                    if res != neblock:
                        raise RuntimeError(f"LZ4 decompress failed: {res} != {neblock}")
                elif compcode == 4:
                    libzstd = _get_libzstd()
                    res = libzstd.ZSTD_decompress(dbuf, neblock, cbuf, len(cbuf))
                    if libzstd.ZSTD_isError(res) or res != neblock:
                        raise RuntimeError(f"ZSTD decompress failed: {res} != {neblock}")
                else:
                    raise RuntimeError(f"Unsupported Blosc compcode: {compcode}")
                block_raw[raw_offset : raw_offset + neblock] = dbuf.raw
            pos += csize
            raw_offset += neblock

        if shuffle and typesize > 1:
            n_elems = bsize // typesize
            if n_elems > 0:
                arr_shuf = np.frombuffer(block_raw, dtype=np.uint8, count=n_elems * typesize).reshape(typesize, n_elems)
                unshuf = arr_shuf.T.tobytes()
                if bsize > n_elems * typesize:
                    unshuf += bytes(block_raw[n_elems * typesize :])
                out[i * blocksize : i * blocksize + bsize] = unshuf
            else:
                out[i * blocksize : i * blocksize + bsize] = block_raw
        elif bitshuffle and typesize > 1:
            out[i * blocksize : i * blocksize + bsize] = unbitshuffle_block(bytes(block_raw), typesize)
        else:
            out[i * blocksize : i * blocksize + bsize] = block_raw

    return bytes(out)
