# LightCone-Spec repository guidance

- Follow [`.codex/project-standards.md`](.codex/project-standards.md) for every change.
- Apply Occam's razor: use the smallest clear implementation that meets the paper protocol.
- Preserve necessary scientific checks; do not add speculative abstractions or blockers.
- Do not start or connect to a GPU host unless the user explicitly authorizes it in the current turn.
- Do not push or publish without explicit user authorization.
- Call results measured only after the configured jobs complete.
