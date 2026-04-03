# /gsd:execute-phase

**But :** Exécuter une phase du workflow via l'exécution parallèle en vagues avec des sous-agents frais.

## Usage
```
/gsd:execute-phase N [flags]
```

## Flags
- `--wave N` — Exécuter uniquement une vague spécifique (pour la gestion de quota)
- `--gaps-only` — Exécuter seulement les plans de comblement de lacunes
- `--interactive` — Exécution séquentielle avec checkpoints utilisateur entre les tâches

## Modèle d'exécution
- Orchestrateur reste "lean" (~15% budget contexte)
- Chaque sous-agent : contexte frais 100%
- Analyse des dépendances → groupement en vagues → spawn parallèle
- Préserve tous les gates : vérification, état, décisions de routage

## Notes importantes
- Les flags ne sont **pas** automatiquement actifs — ils doivent être explicitement présents dans `$ARGUMENTS`
- Sans flag : exécution standard complète de la phase
