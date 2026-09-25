---
name: qwen38-upstream-regression
description: Reproduce the Qwen38 upstream merge regression, including rebuilt wide Q4 projections/FFN, Linux runtime compilation and isolated PR-branch tests. Use after shared GEMV, DMA scheduling or host runtime changes.
---

# Upstream regression of the wide Qwen work

Read `specs/open-engine/plans/upstream-regression.md` and the existing
`open-wide-deltanet-layer/SKILL.md`. The known complete-layer warm-1 precision
gate remains failing; do not confuse it with a new merge regression.

## Rebuild changed kernels

Use ironvenv and XRT as in `open-wide-deltanet-layer`. Run these sequentially:
shared generators overwrite their `.cc` files.

```bash
base=open_kernels/designs/wide_deltanet/build_layer
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 16384 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 6144 --projection-n 5120 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --out open_kernels/designs/layer_x/build_segmented_precise
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-deltanet-layer.py prepare
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py prepare \
  --build-dir open_kernels/designs/layer_x/build_segmented_precise/ffn
```

The complete layer reuses unchanged AB/glue/step/LN/post binaries; build those
as documented by the layer skill if absent. Run both harness cfgs sequentially
with `HARNESS_TIMEOUT_MS=30000` and XRT on the library path, then their
respective `compare` commands. Expected: FFN passes, full layer fails only
`warm-1/y` at0.00908004 versus0.005. Re-prepare after any artifact changes;
never refresh manifest hashes manually to bless stale outputs.

## Host runtime and tests

The installed XRT runtime on this host lacks C++ headers. Matching headers and
linker files are in the existing XRT build:

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
cmake -S src/open_qwen36 -B /tmp/oflm-upstream-qwen -DCMAKE_BUILD_TYPE=Release \
  -DXRT_INCLUDE_DIR="$xrt_sdk/include" -DXRT_LIB_DIR="$xrt_sdk/lib"
cmake --build /tmp/oflm-upstream-qwen -j4
ctest --test-dir /tmp/oflm-upstream-qwen --output-on-failure
ironvenv/bin/python open_kernels/model/replica_block.py --fixture /tmp/oflm-block-fixture
OMP_NUM_THREADS=2 /tmp/oflm-upstream-qwen/block_host_test /tmp/oflm-block-fixture
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Use a matching development SDK on another machine; do not suppress ABI checks.
Standalone CMake now links OpenMP and sets AVX2/FMA for CLI/block-host targets.
Whisper's `host_ops_fast_test` uses setenv/unsetenv on Linux and _putenv_s on
Windows. Compile host/quant Whisper tests with `-mavx2 -mfma`, host-ops with
`-fopenmp`; sources and exact commands are in the report. No new host library
is distributed by this check.

Converter tests require CPU PyTorch, gguf, safetensors, einops and
huggingface-hub in addition to the base test environment. The recorded run used
`pip --target /tmp/oflm-review-deps` and `PYTHONPATH` to avoid modifying ironvenv.
Use `pip --resume-retries 5` for interrupted large wheel downloads. Run
`python -m pytest utilities/q4nx-build/tests -q` in that environment.

For PR checks use separate Git worktrees of qwen38-part1/2/3, with the same
absolute ironvenv Python path. Never change the user's active checkout while
a build or test is reading it. Preserve the original v1 tags as historical
snapshots. Shared fixes belong on each PR branch; full-layer reports do not
belong in PR1/2. Branch publication requires the user's authorization.
