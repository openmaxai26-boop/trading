# /gsd:verify-work

**But :** Valider les fonctionnalités construites via des tests d'acceptation utilisateur (UAT) conversationnels avec état persistant.

## Usage
```
/gsd:verify-work N
```

## Outils disponibles
Read, Bash, Glob, Grep, Edit, Write, Task

## Processus
1. Charge le contexte d'exécution de la phase
2. Exécute les tests un à la fois (réponses texte simple)
3. En cas de problème :
   - Diagnostique les lacunes
   - Planifie les corrections
   - Prépare les étapes d'exécution

## Sortie
Fichier markdown de suivi des résultats UAT pour chaque phase.
Si des défauts sont découverts : problèmes diagnostiqués + plans de correction vérifiés prêts pour `/gsd:execute-phase`.
