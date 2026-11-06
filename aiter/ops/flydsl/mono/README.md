# ML4 decode monokernel (FlyDSL) and the rocKE layer kernel

Mistral Large 4's specialized decode runs each MoE decoder layer as one
persistent launch per TP rank: attention with the fused sparse indexer, the
MoE, and both TP all-reduces inside the kernel. This package and
`aiter.ops.flydsl.kernels.mono` hold that kernel and its host side. vLLM binds
them to the model (weights, caches, the step loop) in
`vllm/models/mistral_large4/amd/specialized_decode/runtime`.

## Packages

| path | contents |
|---|---|
| `mono/op.py`, `mono/variants.py` | the host wrapper (default RoPE variant) and one kernel package per layer kind (RoPE, NoPE, NoPE with KV sharing) |
| `mono/runtime/` | decisions every TP rank takes alike, HIP IPC, the symmetric peer buffer of the in-kernel reductions, the step-begin launch |
| `mono/weights/` | weight storage formats, MXFP4 / MXFP8 helpers, the layer weight container, packing into the kernels' tile layouts |
| `kernels/mono/mistral_large4/gfx950/` | the FlyDSL layer kernel (`kernel.py`), its scratch layout and CTA schedule (`layout.py`, `geometry*.py`), its host op (`op.py`), and `rocke_layer.py` (below) |
| `kernels/mono/common/` | device primitives (wave ops, the tagged mailbox) and build-time contracts (mailbox layout, execution model, build key) |
| `kernels/mono/extern_stages.py`, `sched.py`, `numa.py`, `stamps.py` | rocKE device functions linked into the FlyDSL kernel, schedule levers, NUMA dual-copy mailboxes, stage stamps ([`PHASE23_SUMMARY.md`](../kernels/mono/PHASE23_SUMMARY.md)) |

## The rocKE layer kernel

`AITER_ML4_ROCKE=<rocKE tree>` makes `Glm5MonoKernel` build the rocKE ML4
layer kernel (`library/kernels/gfx950/ml4_mono/layer` in the rocKE tree) for
its configuration and launch it through the HIP module API with the FlyDSL
launcher's arguments unchanged: it places every task on the CTA the FlyDSL
build would and reads and writes the same scratch, peer buffer and caches.
The build runs in a subprocess (FlyDSL embeds its own LLVM) and is cached
under `AITER_ML4_ROCKE_CACHE`, keyed by the spec and the rocKE layer sources;
`AITER_ML4_ROCKE_SPEC` takes the build's schedule knobs as JSON.

`rocke_layer.RockeLauncher` also launches the family's other kinds from a
spec of their own: vLLM's DSpark draft layers (`kind="draft"`,
`vllm/models/mistral_large4/amd/rocke_draft.py`) are built and launched
through it.

## Tests

| file (`op_tests/flydsl_tests/`) | what it checks |
|---|---|
| `test_mono_ml4_layer.py` | GPU: one TP4 rank's layer on an FP8 latent KV cache |
| `test_mono_extern_stages.py` | CPU: the extern-stage ABI, resolver, stage pairing, schedule grammar, NUMA addressing |
| `test_mono_extern_smoke.py` | linking a toy rocKE device function into a FlyDSL kernel |
| `check_rocke_ml4_layer.py` | the rocKE layer kernel against the FlyDSL one, bit for bit, by stage group or the whole layer, on one GPU or TP4 (`--npes 4`) |
| `check_rocke_ml4_draft.py` | the rocKE draft kind against a PyTorch model of vLLM's draft layer (relative error, every stage on its own inputs and the layer end to end), on one GPU or TP4; `--bench` times cold-cache launches, `--timeline` prints the stage timeline |
| `bench_mono_ml4_layer.py` | layer timing, and the weights and state the checks share |

`check_rocke_ml4_*.py` take the rocKE tree as `--rocke` (its `platform/python`
and `library` go on the build process's `PYTHONPATH`).
