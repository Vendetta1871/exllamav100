#!/usr/bin/env python3
# Convert an EXL3-quantized model (safetensors, quant_method == "exl3") to GGUF.
#
# The EXL3 trellis is decoded and the effective dense weight is reconstructed:
#
#     W = diag(suh) @ H @ W_dec @ H @ diag(svh)
#
# where W_dec is the decoded trellis and H is the block-diagonal Hadamard-128
# transform (normalized by 1/sqrt(128)). Decode logic is a port of
# exllamav3_ext/quant/{pack.cu,codebook.cuh,exl3_dq.cuh} from exllamav3.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
)

HAD_DIM = 128

# codebook ids used by exllamav3: 0 = "3inst", 1 = mcg, 2 = mul1
CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2


def _f16(bits: int) -> np.float32:
    return np.array([bits], dtype=np.uint16).view(np.float16).astype(np.float32)[0]


def tensor_core_perm() -> np.ndarray:
    # row-major tile element index for each packed position (quantize.py in exllamav3)
    perm = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        c1 = c0 + 8
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for i in range(4):
            perm[t * 8 + i] = rows[i] * 16 + c0
            perm[t * 8 + 4 + i] = rows[i] * 16 + c1
    return perm


PERM = tensor_core_perm()
PERM_I = np.argsort(PERM)


def unpack_trellis(packed: np.ndarray, K: int) -> np.ndarray:
    # packed: (n_tiles, 16*K) uint16 -> (n_tiles, 256) uint16 codebook indices
    # port of unpack_trellis_kernel (pack.cu)
    n_tiles = packed.shape[0]
    u = packed.reshape(n_tiles, -1, 2).astype(np.uint32)
    words = u[..., 0] | (u[..., 1] << np.uint32(16))
    nwords = 256 * K // 32
    t = np.arange(128, dtype=np.int64)
    b0 = t * 2 * K + K - 16 + 256 * K
    b2 = b0 + K + 16
    i0 = b0 // 32
    i1 = (b2 - 1) // 32
    s1 = ((i1 + 1) * 32 - b2).astype(np.uint64)
    a = words[:, i0 % nwords].astype(np.uint64)
    b = words[:, i1 % nwords].astype(np.uint64)
    # __funnelshift_r(b, a, s) = ((a:b) >> s) & 0xffffffff
    w1 = (((a << np.uint64(32)) | b) >> s1).astype(np.uint32)
    w0 = (w1 >> np.uint32(K)) & np.uint32(0xffff)
    w1 = w1 & np.uint32(0xffff)
    return np.stack((w0, w1), axis=-1).reshape(n_tiles, 256)


def _decode_lop3(x: np.ndarray) -> np.ndarray:
    x = (x & np.uint32(0x8fff8fff)) ^ np.uint32(0x3b603b60)
    lo = (x & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo + hi


def decode_codebook(idx: np.ndarray, cb: int) -> np.ndarray:
    # idx: (n_tiles, 256) uint32 -> float32 weights (codebook.cuh)
    x = idx.astype(np.uint32)
    if cb == CB_3INST:
        x = x * np.uint32(89226354) + np.uint32(64248484)
        return _decode_lop3(x)
    if cb == CB_MCG:
        return _decode_lop3(x * np.uint32(0xCBAC1FED))
    # CB_MUL1
    x = x * np.uint32(0x83DCD12D)
    s = ((x & np.uint32(0xff)) + ((x >> np.uint32(8)) & np.uint32(0xff))
         + ((x >> np.uint32(16)) & np.uint32(0xff)) + (x >> np.uint32(24))
         + np.uint32(0x6400))
    h = (s & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
    return h * _f16(0x1eee) + _f16(0xc931)


def decode_trellis(trellis: torch.Tensor, K: int, cb: int) -> torch.Tensor:
    # trellis: (kb, nb, 16*K) int16 -> W_dec (kb*16, nb*16) float32
    kb, nb, _ = trellis.shape
    packed = trellis.numpy().view(np.uint16).reshape(kb * nb, 16 * K)
    vals = decode_codebook(unpack_trellis(packed, K), cb)
    tiles = vals[:, PERM_I].reshape(kb, nb, 16, 16)
    w_dec = tiles.transpose(0, 2, 1, 3).reshape(kb * 16, nb * 16)
    return torch.from_numpy(np.ascontiguousarray(w_dec))


def unpack_signs(t: torch.Tensor) -> torch.Tensor:
    # legacy packed sign bitfield (exl3.py unpack_bf): int16 -> 16 values of +/-1
    u = t.view(torch.uint16).to(torch.int32)
    bits = (u.unsqueeze(-1) >> torch.arange(16)) & 1
    return (1.0 - 2.0 * bits).flatten().to(torch.float32)


def hadamard_matrix(n: int = HAD_DIM) -> torch.Tensor:
    # sylvester chain from hadamard_1 (util/hadamard.py), normalized
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return torch.from_numpy((h / np.sqrt(n)).astype(np.float32))


HAD_128 = hadamard_matrix()


def reconstruct_weight(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    trellis = parts["trellis"]
    K = trellis.shape[-1] // 16
    cb = CB_MUL1 if "mul1" in parts else CB_MCG if "mcg" in parts else CB_3INST
    suh = parts["suh"].to(torch.float32) if "suh" in parts else unpack_signs(parts["su"])
    svh = parts["svh"].to(torch.float32) if "svh" in parts else unpack_signs(parts["sv"])

    w = decode_trellis(trellis, K, cb)
    k, n = w.shape
    assert suh.numel() == k and svh.numel() == n, \
        f"suh/svh shape mismatch: w {(k, n)}, suh {suh.shape}, svh {svh.shape}"

    # W = diag(suh) @ H @ W_dec @ H @ diag(svh) (exl3.py get_weight_tensor)
    w = (HAD_128 @ w.view(-1, HAD_DIM, n)).view(k, n)
    w *= suh.unsqueeze(1)
    w = (w.view(k, -1, HAD_DIM) @ HAD_128).view(k, n)
    w *= svh.unsqueeze(0)

    # HF/gguf layout is (out_features, in_features)
    return w.T.contiguous().to(torch.float16)


class Exl3Reconstruct:
    # mixin: replaces exl3 tensor groups (trellis/suh/svh/...) with reconstructed dense weights

    EXL3_SUBTENSORS = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "bias")

    def dequant_model(self):
        # exl3 tensors are reconstructed in get_tensors, nothing to dequant here
        pass

    def get_tensors(self):
        groups: dict[str, dict[str, torch.Tensor]] = {}
        passthrough: list[tuple[str, object]] = []
        for name, gen in self.model_tensors.items():
            base, dot, sub = name.rpartition(".")
            if dot and sub in self.EXL3_SUBTENSORS:
                groups.setdefault(base, {})[sub] = gen
            else:
                passthrough.append((name, gen))

        exl3_bases = {base for base, parts in groups.items() if "trellis" in parts}

        for name, gen in passthrough:
            yield name, gen()

        for base, gens in sorted(groups.items()):
            if base not in exl3_bases:
                # stray subtensor without a trellis, pass through untouched
                for sub, gen in gens.items():
                    yield f"{base}.{sub}", gen()
                continue
            parts = {sub: gen() for sub, gen in gens.items()}
            logger.info(f"reconstructing exl3 layer {base} "
                        f"(K = {parts['trellis'].shape[-1] // 16})")
            yield base + ".weight", reconstruct_weight(parts)
            if "bias" in parts:
                yield base + ".bias", parts["bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an EXL3-quantized safetensors model to GGUF (dense F16 reconstruct)")
    parser.add_argument(
        "model", type=Path,
        help="directory containing the EXL3 safetensors model")
    parser.add_argument(
        "--outfile", type=Path, default=None,
        help="path to write to; default: <model>-f16.gguf in the current directory")
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model (for GGUF metadata)")
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel("INFO")

    dir_model: Path = args.model
    if not dir_model.is_dir():
        logger.error(f"Model directory not found: {dir_model}")
        sys.exit(1)

    hparams = ModelBase.load_hparams(dir_model, False)
    quant_method = (hparams.get("quantization_config") or {}).get("quant_method")
    if quant_method != "exl3":
        logger.warning(f"quant_method is {quant_method!r}, expected 'exl3'")

    fname_out = args.outfile or Path(f"./{dir_model.name}-f16.gguf")

    with torch.inference_mode():
        model_architecture = get_model_architecture(hparams, ModelType.TEXT)
        logger.info(f"Model architecture: {model_architecture}")
        try:
            model_class = get_model_class(model_architecture)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        exl3_class = type("Exl3" + model_class.__name__, (Exl3Reconstruct, model_class),
                          {"model_arch": model_class.model_arch})
        if model_class.supports_mtp_export:
            # include MTP layers only if the checkpoint actually has them
            from safetensors import safe_open
            has_mtp = False
            for st in dir_model.glob("*.safetensors"):
                with safe_open(str(st), framework="numpy") as f:
                    if any(n.startswith(("mtp.", "model.mtp.")) for n in f.keys()):
                        has_mtp = True
                        break
            if not has_mtp:
                exl3_class.no_mtp = True
        model_instance = exl3_class(
            dir_model, gguf.LlamaFileType.MOSTLY_F16, fname_out,
            eager=True, model_name=args.model_name,
        )
        logger.info("Exporting model...")
        model_instance.write()
        logger.info(f"Model successfully exported to {model_instance.fname_out}")


if __name__ == "__main__":
    main()
