# Research notes

Output of the `research` skill: questions answered against **primary sources**,
each claim cited. One file per question, named after it.

**This is deliberately not `build_notes.md`.** That file holds what we measured
ourselves, and its whole value is that every number in it was produced on this
machine. Notes here are what the documentation, source or spec *says* — useful,
but a different grade of evidence. Keeping them apart is what lets build_notes be
trusted without checking.

**Graduation.** When something in here is confirmed by our own measurement, move
the confirmed part into `build_notes.md` with the measurement, and leave a
pointer here. That is the only way a claim earns its place there.

Open questions worth the skill, as of 2026-08-21:

- `pip install pynq` over a PetaLinux-built XRT on ZCU102 — no official PYNQ
  image exists for this board, and it is the last unknown step in the chain
  (README §9).
