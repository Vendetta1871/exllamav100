# exllamav100

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with native support
for [EXL3](https://github.com/turboderp-org/exllamav3) quantized models on the
NVIDIA Tesla V100 (sm_70).

exllamav3 itself does not run on the V100: its kernels rely on `mma.m16n8k16`,
`ldmatrix` and `cp.async`, which require sm_75/sm_80+. This fork instead stores
EXL3 weights natively in GGUF and executes them with Volta-compatible kernels
(register/shuffle based GEMV, `wmma` m16n16k16 GEMM).

## What works

- Native EXL3 inference: weights stay in the trellis format end to end, no
  dequantization to F16 at load time.
- Per-layer fractional bitrates (K = 1..8 words per weight, e.g. 2.5 or 3.3 bpw
  mixes) - each tensor carries its own K.
- All three EXL3 codebooks.
- CUDA backend built for sm_70 only, CPU backend for offload (partial `-ngl`
  works).
- MoE models via `GGML_OP_EXL3_MATMUL_ID` (expert-grouped wmma GEMM for
  prefill, split-k GEMV for decode).
- Architectures wired up so far: `llama`, `qwen35`, `gemma4`.
- Everything not needed for V100 + CPU offload was removed (Vulkan, ROCm,
  SYCL, Metal, RPC, other CUDA archs, etc.).

## Performance (Tesla V100 SXM2 16 GB, `-ngl 99`)

| Model                              | pp512 t/s | tg128 t/s |
| ---------------------------------- | --------: | --------: |
| Qwen3.5-4B-exl3-4.00bpw            |     1150  |     97.6  |
| gemma-4-26B-A4B-it-exl3 (MoE, K=2) |      985  |     73.0  |

## Usage

Convert an EXL3 safetensors checkpoint to native GGUF:

```sh
python convert_exl3_to_gguf.py /path/to/exl3-model --native --outfile model-exl3.gguf
```

Build and run (a CUDA toolkit that still supports sm_70 is required):

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/bin/llama-cli -m model-exl3.gguf -ngl 99 -p "The capital of France is" -n 64
```

There is also a non-native mode (`--outtype f16`, default) that reconstructs
dense F16 weights from the trellis; useful as a reference and for small models.

## GGUF layout

Each EXL3 layer is stored as raw tensors next to each other:

- `<prefix>.trellis` (I16, `[16*K, n/16, k/16]`; 4D with an expert axis for MoE)
- `<prefix>.suh` / `<prefix>.svh` (F16 scales/signs)
- optional `<prefix>.bias`
- `exl3.codebook` metadata key selects the codebook

The matmul computes `had128((had128(x * suh)) @ W_dec * svh)` where `had128` is
a normalized 128-point Walsh-Hadamard transform, matching exllamav3 semantics.

## Limitations

- Only sm_70 is targeted; other GPUs/CPUs work but are not a goal.
- LoRA adapters are ignored on EXL3 layers.
- Vision towers of multimodal checkpoints are skipped by the converter.

## License

MIT, same as upstream llama.cpp. See [LICENSE](LICENSE).
