# Project skills

Vendored copies, so they travel with the repo and stay pinned regardless of what
is installed on any given machine.

| skill | what it is for here |
|---|---|
| `research` | primary-source investigation written up as a Markdown note. The open one is `pip install pynq` over the built XRT — there is no official PYNQ image for the ZCU102, and it is the last genuinely unknown step in the chain (README §9). |
| `grilling` | stress-testing a decision before it costs money. Open question 1 — lens FOV and crop — is the live one: it rests on "mid ≈ close" transferring from diverse web photos to centre crops through one fixed lens, which is an assumption, not a measurement. |
| `diagnosing-bugs` | for board bring-up. If `deploy/run_on_board.py` disagrees with the simulation, the cause is somewhere between bitstream, driver and dequantization, and that needs a tight pass/fail loop rather than staring at code. |

## Provenance

Copied 2026-08-21 from the `mattpocock-skills` plugin, version **1.2.3**, at
`~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.2.3/skills/`:

    engineering/research         -> research/
    productivity/grilling        -> grilling/
    engineering/diagnosing-bugs  -> diagnosing-bugs/

`SKILL.md` and `scripts/` are verbatim. Each source directory also carries an
`agents/openai.yaml` — display metadata for a different runtime — which is not
copied because nothing here reads it.

To refresh, diff against that path at whatever version is installed. Nothing in
these files references the plugin root or any sibling skill, so they work
unchanged as project skills.
