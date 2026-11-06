# ML4 rocKE mono Phase 2/3

Implemented in the aiter checkout:

- `extern_stages.py`: ML4's 38-argument rocKE ABI, `rocke_ml4_` symbols,
  variant/path resolution, sidecar pairing checks, file digests, and kernel
  suffixes.
- `sched.py` and `numa.py`: `AITER_MONO_SCHED` grammar/selectors, stage-entry
  holds, NUMA `Box` bitwise views, dual-copy put/poll support, scratch sizing,
  and the once-per-scratch calibration hook.
- `stamps.py`: timing layout/constants stub for future LDS stamp wiring.
- `mistral_large4/gfx950/kernel.py`: router/ug/down dispatch, external
  argument assembly, NUMA-aware mailbox helpers, down epilogue retained in
  FlyDSL, and bounded S>1 `ug` timeline indices.
- `mistral_large4/gfx950/op.py`: NUMA scratch growth and pre-graph
  preparation hook.
- `op_tests/flydsl_tests/test_mono_extern_stages.py`: CPU-only ABI,
  resolver, stage-pairing, schedule grammar, and NUMA address tests.

TODOs:

- Replace the conservative zero-mask calibration fallback with the real XCD
  ping-pong probe when the ML4 runner binding hook is available.
- Wire `stamps.py` into the launcher if LDS per-CTA stamps are preferred over
  the existing timeline buffer.
- Run the extern and NUMA builds on gfx950 and perform the ISA/bitwise gates;
  the current workstation lacks the GPU Python dependencies.

With extern, schedule, and NUMA knobs unset, the original kernel name,
mailbox addresses, stage bodies, and launch path remain selected; no
knob-specific device instructions are emitted by these additions.
