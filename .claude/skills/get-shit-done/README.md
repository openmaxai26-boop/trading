# Get Shit Done (GSD) — Skill

Source: https://github.com/gsd-build/get-shit-done

Système léger et puissant de meta-prompting, context engineering et développement spec-driven pour Claude Code. Automatise le workflow de : initialisation de projet → discussion → planification → exécution → vérification → livraison.

## Problème résolu
Évite le "context rot" — la dégradation de qualité quand les LLMs remplissent leur fenêtre de contexte pendant de longues sessions de développement.

## Installation
```bash
npx get-shit-done-cc@latest
```

## Plateformes supportées
Claude Code, OpenCode, Gemini CLI, Codex, GitHub Copilot CLI, Cursor, Windsurf, Antigravity, Augment

---

## Commandes disponibles (59 total)

### Workflow principal
| Commande | Description |
|----------|-------------|
| `/gsd:new-project` | Initialiser un nouveau projet (questions + recherche + roadmap) |
| `/gsd:discuss-phase N` | Discussion avant planification — capturer décisions |
| `/gsd:plan-phase N` | Créer plan détaillé d'une phase (PLAN.md) |
| `/gsd:execute-phase N` | Exécuter une phase en vagues parallèles avec sous-agents |
| `/gsd:verify-work N` | Valider le travail via UAT conversationnel |
| `/gsd:ship` | Créer PR et livrer le code vérifié |
| `/gsd:autonomous` | Mode autonome : exécute toutes les phases restantes |

### Debug & Review
| Commande | Description |
|----------|-------------|
| `/gsd:debug` | Débogage systématique avec sous-agent isolé (200k tokens) |
| `/gsd:review --phase N` | Revue croisée par d'autres AIs (Gemini, Claude, Codex) |
| `/gsd:forensics` | Analyse forensique du code |
| `/gsd:health` | Diagnostiquer la santé du répertoire `.planning/` |

### Gestion de projet
| Commande | Description |
|----------|-------------|
| `/gsd:progress` | État d'avancement global |
| `/gsd:stats` | Statistiques du projet |
| `/gsd:add-todo` | Ajouter une tâche |
| `/gsd:check-todos` | Vérifier l'état des todos |
| `/gsd:note` | Prendre une note de projet |
| `/gsd:session-report` | Rapport de fin de session |
| `/gsd:pause-work` | Mettre en pause le travail |
| `/gsd:resume-work` | Reprendre le travail |

### Phases & Milestones
| Commande | Description |
|----------|-------------|
| `/gsd:add-phase` | Ajouter une phase au roadmap |
| `/gsd:insert-phase` | Insérer une phase |
| `/gsd:remove-phase` | Supprimer une phase |
| `/gsd:new-milestone` | Créer un nouveau milestone |
| `/gsd:complete-milestone` | Marquer milestone comme terminé |
| `/gsd:milestone-summary` | Résumé d'un milestone |
| `/gsd:audit-milestone` | Auditer un milestone |
| `/gsd:plan-milestone-gaps` | Combler les lacunes d'un milestone |

### Workspaces
| Commande | Description |
|----------|-------------|
| `/gsd:new-workspace` | Créer un workspace |
| `/gsd:list-workspaces` | Lister les workspaces |
| `/gsd:remove-workspace` | Supprimer un workspace |
| `/gsd:workstreams` | Gérer les flux de travail parallèles |

### Utilitaires
| Commande | Description |
|----------|-------------|
| `/gsd:map-codebase` | Cartographier le codebase |
| `/gsd:research-phase N` | Recherche pour une phase |
| `/gsd:validate-phase N` | Valider une phase |
| `/gsd:secure-phase N` | Audit de sécurité d'une phase |
| `/gsd:ui-phase N` | Phase UI spécialisée |
| `/gsd:ui-review` | Revue UI/UX |
| `/gsd:add-tests` | Ajouter des tests |
| `/gsd:audit-uat` | Audit UAT |
| `/gsd:cleanup` | Nettoyer les artefacts |
| `/gsd:docs-update` | Mettre à jour la documentation |
| `/gsd:add-backlog` | Ajouter au backlog |
| `/gsd:review-backlog` | Examiner le backlog |
| `/gsd:profile-user` | Profiler l'utilisateur |
| `/gsd:set-profile` | Configurer le profil |
| `/gsd:settings` | Paramètres GSD |
| `/gsd:update` | Mettre à jour GSD |
| `/gsd:help` | Aide |
| `/gsd:do` | Exécuter une action rapide |
| `/gsd:fast` | Mode rapide |
| `/gsd:quick` | Action rapide |
| `/gsd:next` | Prochaine étape recommandée |
| `/gsd:thread` | Gérer les threads |
| `/gsd:manager` | Accès au gestionnaire |
| `/gsd:plant-seed` | Planter une graine (idée future) |
| `/gsd:reapply-patches` | Réappliquer des patches |
| `/gsd:pr-branch` | Créer branche PR |
| `/gsd:join-discord` | Rejoindre Discord GSD |
| `/gsd:list-phase-assumptions` | Lister les hypothèses d'une phase |

---

## Artefacts produits
```
.planning/
├── PROJECT.md          — Contexte principal du projet
├── config.json         — Paramètres du workflow
├── STATE.md            — Mémoire/état du projet
├── REQUIREMENTS.md     — Exigences du projet
├── ROADMAP.md          — Plan de phases
└── research/           — Recherches de domaine
```

## Modèle d'exécution
- Orchestrateur léger (~15% budget contexte)
- Chaque sous-agent : contexte frais 100% (200k tokens)
- Exécution en vagues parallèles avec gestion de dépendances
- Commits git atomiques par tâche complétée
