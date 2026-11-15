# cubecl 0.11.4 + per-lane matrix fragments

Branch `cubecl-0.11.4-fragments`: tag `v0.11.4` plus a backport of upstream
[#1858](https://github.com/tracel-ai/cubecl/pull/1858) ("feat(metal): the simdgroup matrix as a
manual MMA", merged to `main` as `adf23788`, not in any 0.11 release) and one test.

## Why

A Metal attention or GEMM kernel written in CubeCL needs to touch its own fragment elements (the
softmax's row max, `exp`, the per-row rescale `o = f o + p v`) without a round trip through
threadgroup memory, as MLX does with `simdgroup_matrix::thread_elements()`. In 0.11.4 the Metal
backend has `cmma::Matrix` (opaque) but not the manual MMA, whose registers are plain per-unit
arrays.

## What

Nothing new: upstream's manual MMA API, now lowered on Metal.

- `MmaDefinition::<A, B, CD>::new(8, 8, 8)`; f16/f16/f16, f16/f16/f32, bf16/bf16/bf16 and
  f32/f32/f32, registered as `features.matmul.mma` on the native Metal runtime and on wgpu with MSL.
- Each unit holds its share in `Array<Vector<T, 2>>` (`elems_per_lane = 2`, `vector_size = 2`,
  one vector): read and write it like any array.
- `position_of_nth(lane, i, ident)` gives each element's `(row, col)` (`RowIndexOp`/`ColIndexOp`
  lowered to the layout below).
- `execute` / `execute_inplace` lower to `simdgroup_multiply_accumulate`, moving the registers in
  and out of `simdgroup_matrix` through `thread_elements()`; the Metal compiler keeps them in
  registers.

CUDA has had the same API since upstream #810 (`mma.sync` register fragments,
`position_of_nth` = the PTX fragment layouts), so a kernel written against `MmaDefinition` and
`position_of_nth` runs on both; only the shapes differ (Metal 8x8x8, CUDA 16x8x{8,16,32}).
`cmma::Matrix` stays opaque on both.

## Per-lane layout (Metal 8x8, every operand: A, B with rows along k, accumulator)

Lane `l` (0..32) holds element `i` (0..2) at

    row = (l / 16) * 4 + (l / 2) % 4
    col = ((l / 8) % 2) * 4 + (l % 2) * 2 + i

so two adjacent columns of one row. A row's 8 cells are held by the 4 lanes that differ in lane
bits 0 and 3: a row reduction is two `plane_shuffle_xor`s, masks 1 and 8. Apple does not document
the layout; it is MLX's `BaseMMAFrag<T, 8, 8>::get_coord`, and the tests check it on the device.

## Backport adaptations

Upstream #1858 cherry-picked with `-x`; conflicts only where it touches code 0.11.4 lacks:

- `cubecl-wgpu/src/compiler/base.rs`: added `WgpuCompiler::target_properties` (and
  `AutoCompiler`'s) without upstream's earlier `limit_properties` method, which 0.11.4 lacks.
- `cubecl-wgpu/src/runtime.rs`: `let compiler = C::init(..)` kept, without the
  `limit_properties` call.
- `cubecl-core/src/runtime_tests/cmma.rs`: took `test_cmma_manual_places_every_cell`, not the FP8
  test (`test_cmma_manual_fp8_matches_cpu_reference`) that sat next to it upstream.

## Tests (M3 Max, native Metal runtime)

`cargo test -p cubecl-metal --lib cmma_manual`:

- `test_cmma_manual` (upstream; Metal shapes added by #1858)
- `test_cmma_manual_places_every_cell` (upstream #1858)
- `test_cmma_manual_row_rescale` (this fork): one 8x8x8 product, each unit reads its accumulator
  cells, takes its rows' max over the plane, computes `scale[row] * x - rowmax`, writes back, and a
  second MMA accumulates onto the result; exact against the CPU for f32/f32, f16/f32, f16/f16 and
  bf16/bf16.

## CUDA: the stream with `get_resource`

`client.get_resource::<CudaServer>(handle)` also fills `GpuResource::stream`: the client's current
stream (the one its launches go to, already waiting on the buffer's own stream). Native work queued
there (sdp's cuBLAS GEMMs) stays in order with CubeCL's launches, no sync.
