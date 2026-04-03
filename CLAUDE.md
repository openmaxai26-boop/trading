# CLAUDE.md — Instructions de Session

## DÉMARRAGE DE SESSION
1. Lire `tasks/lessons.md` — appliquer toutes les leçons avant de toucher quoi que ce soit
2. Lire `tasks/todo.md` — comprendre l'état actuel
3. Si aucun des deux n'existe, les créer avant de commencer

## SKILLS DISPONIBLES
Les compétences suivantes sont disponibles dans `.claude/skills/` :

| Skill | Description | Dossier |
|-------|-------------|---------|
| **get-shit-done** | Système de meta-prompting et context engineering | `.claude/skills/get-shit-done/` |
| **awesome-claude-code** | Catalogue de slash-commands, agent skills, templates | `.claude/skills/awesome-claude-code/` |
| **ui-ux-pro-max** | Design intelligence : 67 styles UI, 161 palettes, 57 typos | `.claude/skills/ui-ux-pro-max/` |
| **awesome-claude-code-subagents** | 130+ sous-agents spécialisés par catégorie | `.claude/skills/awesome-claude-code-subagents/` |

## WORKFLOW

### 1. Planifier d'abord
- Passer en mode plan pour toute tâche non triviale (3+ étapes)
- Écrire le plan dans `tasks/todo.md` avant d'implémenter
- Si quelque chose ne va pas, STOP et re-planifier — ne jamais forcer

### 2. Stratégie sous-agents
- Utiliser des sous-agents pour garder le contexte principal propre
- Une tâche par sous-agent
- Investir plus de compute sur les problèmes difficiles
- Référencer `.claude/skills/awesome-claude-code-subagents/` pour choisir le bon sous-agent

### 3. Boucle d'auto-amélioration
- Après toute correction : mettre à jour `tasks/lessons.md`
- Format : `[date] | ce qui a mal tourné | règle pour l'éviter`
- Relire les leçons à chaque démarrage de session

### 4. Standard de vérification
- Ne jamais marquer comme terminé sans preuve que ça fonctionne
- Lancer les tests, vérifier les logs, comparer le comportement
- Se demander : « Est-ce qu'un staff engineer validerait ça ? »

### 5. Exiger l'élégance
- Pour les changements non triviaux : existe-t-il une solution plus élégante ?
- Si un fix semble bricolé : le reconstruire proprement
- Ne pas sur-ingénieriser les choses simples

### 6. Correction de bugs autonome
- Quand on reçoit un bug : le corriger directement
- Aller dans les logs, trouver la cause racine, résoudre
- Utiliser `/gsd:debug` si disponible (voir `.claude/skills/get-shit-done/commands/debug.md`)

## PRINCIPES FONDAMENTAUX
- Simplicité d'abord — toucher un minimum de code
- Pas de paresse — causes racines uniquement, pas de fixes temporaires
- Ne jamais supposer — vérifier chemins, APIs, variables avant utilisation
- Demander une seule fois — une question en amont si nécessaire, ne jamais interrompre en cours de tâche

## GESTION DES TÂCHES
1. Planifier → `tasks/todo.md`
2. Vérifier → confirmer avant d'implémenter
3. Suivre → marquer comme terminé au fur et à mesure
4. Expliquer → résumé de haut niveau à chaque étape
5. Apprendre → `tasks/lessons.md` après corrections

## COMMANDES GSD DISPONIBLES
Voir `.claude/skills/get-shit-done/README.md` pour la liste complète.

Commandes clés :
- `/gsd:new-project` — Initialiser un nouveau projet
- `/gsd:plan-phase N` — Planifier une phase
- `/gsd:execute-phase N` — Exécuter une phase
- `/gsd:verify-work N` — Vérifier le travail
- `/gsd:debug` — Déboguer un problème
- `/gsd:ship` — Créer un PR et livrer
- `/gsd:autonomous` — Mode autonome complet

## DESIGN UI/UX
Pour toute tâche UI/UX, consulter `.claude/skills/ui-ux-pro-max/SKILL.md`.

Commande de recherche :
```bash
python3 src/ui-ux-pro-max/scripts/search.py "<query>" --domain <domain>
```
