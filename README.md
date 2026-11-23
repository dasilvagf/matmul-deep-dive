# 🔢 Matrix Multiplication Deep Dive

> Every rung of the GEMM ladder, from scalar C++ to Tensor Cores and RDNA4 WMMA, measured down to the SASS and RDNA ISA. Boldly going where cuBLAS has gone before! 🖖

This repository is the code companion to the **Matrix Multiplication Deep Dive** blog series on [dasilvagf.github.io](https://dasilvagf.github.io). Each step of the ladder is a self-contained kernel, measured against the hardware peak and the vendor library, and explained down to the instruction level.

The goal is not only *"how fast?"* but *"why exactly this fast, and what is the next bottleneck?"*

---

## 📚 Blog Series

| # | Post | Hardware | Status |
|---|------|----------|--------|
| 0 | Methodology: how I measure, validate and read a roofline | — | ⏳ |
| 1 | CPU: SIMD, threads and why Strassen loses | i7 | ⏳ |
| 2 | GPU SIMT Ladder I: memory | RTX 2070 Max-Q | ⏳ |
| 2.5 | Inside a Turing SM: microbenchmarks | RTX 2070 Max-Q | ⏳ |
| 3 | GPU SIMT Ladder II: compute | RTX 2070 Max-Q | ⏳ |
| 4 | Tensor Cores on Turing | RTX 2070 Max-Q | ⏳ |
| 5 | Inference shapes: GEMV, split-K, fused epilogues | RTX 2070 Max-Q | ⏳ |
| 6 | Quantization: INT8 and INT4 weight-only | RTX 2070 Max-Q | ⏳ |
| 7 | Same kernel, two vendors (ISA level) | + RX 9060 XT | ⏳ |
| 8 | Triton: what you get for free | Cloud | ⏳ |
| 9 | Hopper: wgmma, TMA, warp specialization | Cloud | ⏳ |
| 10 | Series conclusion | All | ⏳ |

✅ published · 🚧 in progress · ⏳ planned

---

## 🖥️ Hardware

| Rung | GPU | Architecture | Role |
|------|-----|--------------|------|
| 1 | RTX 2070 Max-Q 8 GB | Turing (sm_75) | Main development target |
| 2 | RX 9060 XT 16 GB | RDNA4 (gfx1200) | AMD / HIP / cross-vendor |
| 3 | Cloud (Ampere, Ada, Hopper) | sm_86 / sm_89 / sm_90 | Final benchmarks only |
| — | Intel Core i7 | x86-64 (Skylake family) | CPU reference |

A full hardware sheet (clocks, peak FLOP/s per data type, peak bandwidth) lives in [`docs/hardware.md`](docs/hardware.md).

---

## 📏 Methodology

- **Correctness:** every kernel is validated against an FP64 CPU reference (small and medium sizes) or cuBLAS / rocBLAS (large sizes, low precision), with a tolerance per data type.
- **Timing:** warmup runs, median of N, `cudaEvent` / `hipEvent`.
- **Clocks:** locked when possible. On the laptop (Max-Q) GPU, clock and temperature are monitored during each run, with cooldown between runs, and the **real measured clock** is reported.
- **Metrics:** GFLOP/s (GB/s for GEMV), % of peak at the measured clock, % of the vendor library.
- **Roofline** plot per kernel.
- **Every step is tracked:** registers per thread, spills, occupancy and the generated ISA.

Each optimization follows the same template:

> 🎯 **Hypothesis** · 📈 **Measured** (profiler sections) · 🔍 **ISA proof** (SASS / RDNA ISA) · 📊 **Result**

### Matrix shapes

| Category | Shapes (M×N×K) |
|----------|----------------|
| Square | 256³, 1024³, 4096³ |
| Rectangular | 1000×10000×1000 |
| Inference | 1×4096×4096, 1×11008×4096 (GEMV decode), 16×4096×4096, 64×4096×4096 |
| Ugly sizes | 1000³, 4097³ |

---

## 🪜 The Ladder

### CPU
| Kernel | Status |
|--------|--------|
| Naive (also the FP64 validation reference) | ✅ |
| SIMD | ✅ |
| Multithreaded | ✅ |
| Strassen | ✅ |

### GPU – SIMT (FP32)
| Kernel | Status |
|--------|--------|
| Naive | ✅ |
| Coalesced global access | ⏳ |
| Tiled (shared memory) | ✅ |
| Register blocking (1D / 2D thread tile) | ⏳ |
| Vectorized loads (`float4`) | ⏳ |
| Bank conflict fix (padding vs swizzle) | ⏳ |
| Warp tiling | ⏳ |
| Double buffering | ⏳ |
| Autotuning per shape | ⏳ |

### GPU – Tensor Cores (FP16, FP32 accumulate)
| Kernel | Status |
|--------|--------|
| WMMA | ✅ |
| `mma.sync` + `ldmatrix` (inline PTX) | ⏳ |
| Swizzled shared memory | ⏳ |
| Register-staged pipeline | ⏳ |

### Inference & Quantization
| Kernel | Status |
|--------|--------|
| GEMV (M = 1 decode) | ⏳ |
| Split-K | ⏳ |
| Epilogue fusion (bias + activation + dequant) | ⏳ |
| INT8 GEMM (`dp4a`, `mma.sync`) | ⏳ |
| INT4 weight-only GEMM / GEMV | ⏳ |

### AMD (RDNA4, HIP)
SIMT port, WMMA (FP16 / BF16 / FP8 / INT8), inference kernels and a side-by-side ISA comparison. ⏳

---

## 📊 Results

Results are added as each blog post is published, always with the measured clock, % of peak and % of the vendor library.

*(Coming with Part 1.)*

---

## 🔧 Building

**Requirements**
- CMake 3.24+
- CUDA Toolkit (for NVIDIA targets)
- ROCm / HIP (for AMD targets, Linux)
- A C++20 compiler

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Test data** is generated locally, not stored in the repository:

```bash
# example, see tools/ for the actual script
python tools/generate_matrices.py --out data/
```

**Profiling:** Nsight Compute / Nsight Systems on NVIDIA, rocprof / Omniperf / RGP on AMD.

---

## 📁 Repository Layout

```
include/   shared headers (timing, validation, CPU utilities)
src/cpu/   CPU kernels
src/cuda/  NVIDIA kernels
src/hip/   AMD kernels
tools/     data generation, benchmarking and plotting scripts
docs/      hardware sheet, methodology notes, roofline data
```

---

## ⚖️ Disclaimer

This is a personal study project. It is not affiliated with any employer and is built only from public documentation, public tools and my own measurements.

## 📜 License

MIT. See [LICENSE](LICENSE).

---

**Gabriel da Silva** · [Blog](https://dasilvagf.github.io) · [LinkedIn](https://www.linkedin.com/in/gfelipess7) · Live long and prosper! 🖖
