# /gsd:discuss-phase

**But :** Phase de discussion pour clarifier les décisions d'implémentation avant la planification.

## Usage
```
/gsd:discuss-phase N [--auto] [--chain]
```

## Flags
- `--auto` — Ignore les étapes interactives
- `--chain` — Enchaîne avec la prochaine commande

## Processus
1. Charge le contexte de projet existant
2. Scanne le codebase pour les patterns réutilisables
3. Identifie les "zones grises" (questions non résolues)
4. Permet à l'utilisateur de sélectionner les zones à explorer
5. Génère `CONTEXT.md` avec les décisions verrouillées

## Sortie
`CONTEXT.md` — Capture des décisions critiques pour les agents en aval
