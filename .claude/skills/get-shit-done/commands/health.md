# /gsd:health

**But :** Diagnostiquer l'intégrité du répertoire `.planning/` et optionnellement réparer les problèmes.

## Usage
```
/gsd:health [--repair]
```

## Flags
- `--repair` — Répare automatiquement les problèmes découverts

## Outils disponibles
Read, Bash, Write, AskUserQuestion

## Vérifications effectuées
- Fichiers manquants
- Configurations invalides
- État incohérent
- Plans orphelins
