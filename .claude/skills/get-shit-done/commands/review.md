# /gsd:review

**But :** Demander une revue cross-AI des plans de phase et générer REVIEWS.md.

## Usage
```
/gsd:review --phase N [--gemini] [--claude] [--codex] [--opencode] [--all]
```

## Options
- `--phase N` — Phase à reviewer (requis)
- `--gemini` — Inclure la revue Gemini CLI
- `--claude` — Inclure la revue Claude CLI (session séparée)
- `--codex` — Inclure la revue Codex CLI
- `--opencode` — Inclure la revue OpenCode
- `--all` — Inclure tous les reviewers disponibles

## Outils disponibles
Read, Write, Bash, Glob, Grep

## Sortie
`REVIEWS.md` — Feedback consolidé de tous les reviewers.
Peut être réintégré dans la planification via `--reviews` flag.
