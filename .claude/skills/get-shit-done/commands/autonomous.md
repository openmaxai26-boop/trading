# /gsd:autonomous

**But :** Exécuter automatiquement toutes les phases restantes du projet (discuss → plan → execute pour chaque phase).

## Usage
```
/gsd:autonomous [flags]
```

## Flags
- `--from N` — Reprendre depuis une phase spécifique
- `--only N` — Exécuter uniquement la phase N
- `--interactive` — Questions utilisateur en ligne avec dispatch plan/execute en arrière-plan

## Outils disponibles
Read, Write, Bash, Glob, Grep, AskUserQuestion, Task

## Gestion d'état
Mise à jour automatique de :
- `.planning/STATE.md`
- `.planning/ROADMAP.md`
- Artefacts de phase (CONTEXT.md, PLANs, SUMMARYs)

## Flux de complétion
Découverte de phase → Exécution → Audit → Nettoyage final

## Pause automatique
Ne s'arrête que pour les décisions contentieuses ou les blockers nécessitant validation utilisateur.
