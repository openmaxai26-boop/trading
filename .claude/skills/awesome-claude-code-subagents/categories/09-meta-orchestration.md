# Catégorie 09 — Meta & Orchestration

Plugin: `voltagent-meta` | 13 agents

Coordination avancée d'agents et automatisation de workflows.

## Agents disponibles

| Agent | Spécialité |
|-------|-----------|
| **orchestrator** | Coordination multi-agents, délégation de tâches |
| **context-optimizer** | Optimisation du contexte, compression d'information |
| **workflow-automator** | Automatisation de workflows répétitifs |
| **agent-supervisor** | Supervision et validation des sorties d'agents |
| **task-decomposer** | Décomposition de tâches complexes |
| **parallel-executor** | Exécution parallèle de tâches indépendantes |
| **dependency-resolver** | Résolution de dépendances entre tâches |
| **result-synthesizer** | Synthèse des résultats de multiples agents |
| **quality-gatekeeper** | Validation qualité avant chaque étape |
| **session-manager** | Gestion de sessions longues et mémoire |
| **airis-mcp-gateway** | Intégration AIRIS MCP Gateway |
| **taskade-integrator** | Intégration Taskade |
| **meta-planner** | Planification de workflows multi-agents |

## Quand utiliser
- Projets complexes nécessitant multiple agents
- Optimisation des coûts de tokens
- Workflows entièrement automatisés
- Supervision de la qualité en pipeline

## Pattern recommandé
```
orchestrator
  ├── task-decomposer (analyser les dépendances)
  ├── parallel-executor (tâches indépendantes)
  ├── quality-gatekeeper (validation)
  └── result-synthesizer (consolidation)
```
