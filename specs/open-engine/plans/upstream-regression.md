# Upstream merge review — 2026-09-25–26

Reviewed `8e05061`, the remote merge of upstream through `eb65600` into
`implement-qwen38-27b-support`, against our preceding `bbefb79`. The user had
already pulled that integration branch. Their rebased PR branches are
`qwen38-part1/2/3`; their existing commits were patch-equivalent to our original
PR snapshots. Original v1 tags and `upstream-pr/*` snapshot branches are retained.

## Overlap review

Seven paths were edited on both sides: `layer_x/lx.py`, `xcommon.py`,
`recipes/dense.py`, the canonical spec, and the C++ manifest header/parser/test.
The merge preserves wide activation streaming, banked AB, segmented FFN,
precise wide LN and the complete-layer fixture.

- Upstream #113 changes DeltaNet transfer issue order from core-major to
  head-major and adds `Pipeline.finish_oldest`. New regression tests cover
  all32/48 heads, exact state/output offsets, per-endpoint record/state order,
  and waiting for QKV without accidentally waiting for Z or another channel.
- Shared Q4 GEMV replaces runtime division/remainder by shifts/masks for the
  supported row splits2/4. Rebuilt both full-width projections and segmented
  FFN, retaining the existing reference and tolerances.
- Gemma dense prefill additions coexist with the FFN and norm cache changes.
  Updated manifests parse; sequential attention-stream misrouting is refused.
- Runtime #112/#114 adds router sentinels and same-context submit-ahead.
  Reviewed dispatch ordering, shared instruction-BO lifetime, explicit waits
  at context boundaries, and the serial deepstack path. Runtime compiles and
  its host/manifest tests pass. No real-model submit-ahead benchmark was run.
- Whisper, converter and packaging changes are largely independent of our
  geometry work. Whisper host arithmetic, generation and guards were tested.

No corrective change to the merged tensor layouts or NPU arithmetic was
needed. Two Linux build/test issues were found and fixed in `623f4bc`:

1. Standalone `open_qwen36` omitted AVX2/FMA and OpenMP for targets compiling
   `block_host.cpp`. Compilation first failed on intrinsics; after flags were
   supplied, linking exposed missing OpenMP. CMake now declares both requirements.
   This is an inherited standalone-build issue, not a tensor-layout merge bug.
2. The incoming Whisper host-fast default test unconditionally used Windows
   `_putenv_s`. It now uses POSIX setenv/unsetenv on Linux. The same test failed
   compilation before the change and passes afterwards.

## Verification

| Check | Result |
| --- | --- |
| Integration open-engine | 759 passed,47 skipped |
| PR1 open-engine | 694 passed,47 skipped |
| PR2 open-engine | 724 passed,47 skipped |
| PR3 open-engine | 748 passed,47 skipped |
| Converter | 81 passed,38 subtests passed |
| Standalone C++ runtime | Release build succeeds |
| CTest: manifest + vision config/window | 3/3 pass |
| Block host against generated independent fixture | PASS |
| Whisper host-fast, quantization, HF-generation, guards | All pass |

PRs were checked in isolated worktrees, using the same ironvenv. Missing
PyTorch/gguf dependencies were installed into `/tmp/oflm-review-deps`, without
changing ironvenv. The first converter collection failure was environmental.
Server/tool-calling tests collected but skipped58 checks without a running
server; this is not end-to-end application acceptance. Optional real Whisper
weight checks also remain skipped without a model container.

## Hardware and limitations

Rebuilt K5120/N16384 QKV+Z, K6144/N5120 output projection, and H5120/FF17408
segmented FFN from the merged shared GEMV/xcommon sources. Unchanged AB,
convolution/recurrence, LN and post binaries were reused. The open harness
completed90 complete-layer dispatches and13 standalone FFN dispatches with
no timeout. This tests the synthetic NPU composition, not the runtime's
submit-ahead execution of a real64-layer model. The fused wide DMA guard and
model/catalogue gates remain in place.

The previous full-layer accuracy failure must remain visible; source and seed
are unchanged. See the final measured results below and the earlier
[complete-layer report](wide-deltanet-layer.md). These results do not authorize
calling the 27B model supported.

Exact kernel build and test workflow:
[qwen38-upstream-regression skill](../../../.opencode/skill/qwen38-upstream-regression/SKILL.md).
Host uses Python3.14.4, mlir-aie1.4.2, llvm-aie21.0.0.2026080301+c9c5ecb7,
XRT2.26.0, Ryzen AI9 365/Strix. CPU converter dependencies include
PyTorch2.14.0+cpu. Logs are retained under the ignored
`open_kernels/designs/wide_deltanet/build_layer/upstream-review/` directory.

Whisper host tests on Linux, from repository root:

```bash
g++ -std=c++17 -O2 -mavx2 -mfma -fopenmp -Isrc -Isrc/include \
  src/open_whisper/host_ops_fast_test.cpp src/open_whisper/host_ops.cpp \
  src/open_qwen36/q4nx_file.cpp -o /tmp/oflm-whisper-host-test
OMP_NUM_THREADS=2 /tmp/oflm-whisper-host-test
g++ -std=c++17 -O2 -mavx2 -mfma -Isrc -Isrc/include \
  src/open_whisper/decoder_quant_test.cpp src/open_whisper/decoder_quant.cpp \
  src/open_qwen36/q4nx_file.cpp -o /tmp/oflm-whisper-quant-test
/tmp/oflm-whisper-quant-test
g++ -std=c++17 -O2 -Isrc/include src/open_whisper/generation_hf_test.cpp \
  src/common/whisper/generation_hf.cpp -o /tmp/oflm-whisper-generation-test
/tmp/oflm-whisper-generation-test src/open_whisper/testdata
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
g++ -std=c++17 -O2 -Isrc -Isrc/include -Isrc/open_npue -I"$xrt_sdk/include" \
  src/open_whisper/guards_test.cpp src/open_whisper/guards.cpp \
  src/open_qwen36/q4nx_file.cpp -o /tmp/oflm-whisper-guards-test
/tmp/oflm-whisper-guards-test /tmp
```

## Final measurements and branch updates

The entire serialized layer result (all874 checks and diagnostics) is identical
before/after upstream. Exactly one check remains failing: warm-1/y,
maxrel0.009080040153299307. All13 standalone FFN inputs pass (26 tensor checks);
strict tolerances and the input seed were unchanged.

Shared fix commits tested on the user-created PR branches:

| Branch | Fix commit |
| --- | --- |
| implement-qwen38-27b-support | 623f4bc |
| qwen38-part1 | 708c4ae |
| qwen38-part2 | b1c9c17 |
| qwen38-part3 | 2fd6072 |

SHA256 of the regenerated manifests and measured results:

```text
7bb764d23b03508f1cb3c310167959232926e90e85ff16ee18d08c1baf69d0cc  open_kernels/designs/wide_deltanet/build_layer/acceptance/layer-fixture.json
d3fa3f209652edeb3c46d6989557d9ce3be0c5ff3a08785dfc3d83f10fa3b182  open_kernels/designs/wide_deltanet/build_layer/acceptance/layer-results.json
79d3e5c184f0ddf47f1b2451ef6bc77db359700cb00b0e1252aa0f6d97cf2234  open_kernels/designs/layer_x/build_segmented_precise/ffn/segmented-results.json
7caba68b78f68da1b655edd7e5aea57e34bb4b0be81ba9f5d03690c29c4ec6b8  open_kernels/designs/wide_deltanet/build_layer/projection_k5120/final.xclbin
b96af535e58b825d4a589aabf6917c38078261acd273353a51a33d2ba1ce32f8  open_kernels/designs/wide_deltanet/build_layer/projection_k6144/final.xclbin
856658a8a5abeeebf08263320e37aeb8a12acede01c50ace64b06ab3109f9967  open_kernels/designs/layer_x/build_segmented_precise/ffn/final.xclbin
```
