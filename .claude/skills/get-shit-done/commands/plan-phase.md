# /gsd:plan-phase

**But :** Créer des plans de phase détaillés (fichiers PLAN.md) avec recherche intégrée et boucles de vérification.

## Usage
```
/gsd:plan-phase N [flags]
```

## Flags
- `--research` — Force une nouvelle recherche même si elle existe déjà
- `--skip-research` — Ignore la recherche entièrement
- `--gaps` — Mode comblement de lacunes (utilise les données de vérification existantes)
- `--skip-verify` — Ignore la boucle de vérification
- `--prd <fichier>` — Utilise un fichier PRD au lieu de la phase de discussion
- `--reviews` — Replanifie en incorporant les retours cross-AI
- `--text` — Utilise des listes en texte brut (requis pour les sessions distantes)

## Flux par défaut
Research → Plan → Verify → Done

## Outils disponibles
Read, Write, Bash, Glob, Grep, Task, WebFetch

## Sortie
Fichier `PLAN.md` pour la phase N dans `.planning/`
