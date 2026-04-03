# /gsd:new-project

**But :** Initialiser un nouveau projet avec un workflow structuré combinant questions, recherche, extraction d'exigences et création de roadmap.

## Usage
```
/gsd:new-project [--auto]
```

## Flags
- `--auto` — Mode automatique, exécute le pipeline complet sans pauses

## Processus
1. Questions sur le projet et ses objectifs
2. Recherche de domaine (si pertinent)
3. Extraction des exigences
4. Création du roadmap par phases

## Artefacts créés
- `.planning/PROJECT.md` — Contexte principal du projet
- `.planning/config.json` — Paramètres du workflow
- `.planning/research/` — Recherches optionnelles
- `.planning/REQUIREMENTS.md` — Exigences scoped du projet
- `.planning/ROADMAP.md` — Plan d'exécution par phases
- `.planning/STATE.md` — Mémoire/état de suivi

## Étape suivante
Après complétion, lancer `/gsd:plan-phase 1`
