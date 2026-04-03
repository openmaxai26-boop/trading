# Awesome Claude Code Subagents

Source: https://github.com/VoltAgent/awesome-claude-code-subagents

Collection de **130+ sous-agents Claude Code spécialisés** organisés en 10 catégories. Chaque sous-agent fonctionne avec une fenêtre de contexte indépendante pour éviter la contamination croisée entre tâches.

## Avantages des sous-agents
- Contexte indépendant par tâche
- Expertise spécifique au domaine
- Partageables entre projets
- Contrôle granulaire des permissions d'outils
- Routage automatique vers les modèles Claude appropriés (Opus/Sonnet/Haiku)

## Installation

```bash
# Via Claude Code Plugin (recommandé)
claude plugin install voltagent-core-dev

# Manuel
cp agent-file.md ~/.claude/agents/

# Script interactif
./install-agents.sh

# Standalone (sans clone)
curl -sSL <install-url> | bash
```

## 10 Catégories

| # | Plugin | Agents | Domaine |
|---|--------|--------|---------|
| 01 | `voltagent-core-dev` | 11 | Développement essentiel quotidien |
| 02 | `voltagent-lang` | 30+ | Spécialistes par langage |
| 03 | `voltagent-infra` | 16 | Infrastructure & DevOps |
| 04 | `voltagent-qa-sec` | 15 | Qualité & Sécurité |
| 05 | `voltagent-data-ai` | 13 | Data & IA |
| 06 | `voltagent-dev-exp` | 14 | Expérience développeur |
| 07 | `voltagent-domains` | 12 | Domaines spécialisés |
| 08 | `voltagent-biz` | 11 | Business & Produit |
| 09 | `voltagent-meta` | 13 | Orchestration multi-agents |
| 10 | `voltagent-research` | 3 | Recherche & Analyse |

Voir les fichiers dans `categories/` pour le détail de chaque catégorie.
