# UI/UX Pro Max — Guide d'intégration Claude

Source: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill

## Vue d'ensemble du projet

Antigravity Kit est un toolkit de design intelligence alimenté par l'IA fournissant des bases de données consultables de styles UI, palettes de couleurs, pairings de polices, types de graphiques et guidelines UX. Fonctionne comme un skill/workflow pour les assistants IA (Claude Code, Windsurf, Cursor, etc.).

## Architecture
```
src/ui-ux-pro-max/                # Source de vérité
├── data/                         # Bases de données CSV canoniques
│   ├── products.csv, styles.csv, colors.csv, typography.csv, ...
│   └── stacks/                   # Guidelines spécifiques par stack
├── scripts/
│   ├── search.py                 # Point d'entrée CLI
│   ├── core.py                   # Moteur de recherche hybride BM25 + regex
│   └── design_system.py          # Génération de design system
└── templates/
    ├── base/                     # Templates de base (skill-content.md, quick-reference.md)
    └── platforms/                # Configs par plateforme (claude.json, cursor.json, ...)
```

## Moteur de recherche
Utilise le ranking BM25 combiné avec la correspondance regex. Détection automatique de domaine disponible quand `--domain` est omis.

## Règles de sync
**Source de vérité :** `src/ui-ux-pro-max/`

Lors de modifications :
1. **Data & Scripts** — Éditer dans `src/ui-ux-pro-max/` (auto-disponible via symlinks)
2. **Templates** — Éditer dans `src/ui-ux-pro-max/templates/`
3. **CLI Assets** — Synchroniser avant publication :
   ```bash
   cp -r src/ui-ux-pro-max/data/* cli/assets/data/
   cp -r src/ui-ux-pro-max/scripts/* cli/assets/scripts/
   cp -r src/ui-ux-pro-max/templates/* cli/assets/templates/
   ```

## Prérequis
Python 3.x (aucune dépendance externe requise)
