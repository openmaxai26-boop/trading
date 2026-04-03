# UI/UX Pro Max — Design Intelligence Skill

Source: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
Version: 2.5.0 | Auteur: NextLevelBuilder | Licence: MIT

Système d'intelligence design alimenté par l'IA fournissant des recommandations UI/UX professionnelles pour web et mobile.

## Quand utiliser ce skill
Déployer pour toute tâche impliquant :
- Structure UI et décisions de design visuel
- Patterns d'interaction et expérience utilisateur
- Assurance qualité UX
- Tout ce qui affecte "comment ça looks, feels, moves, ou est interagi"

## Installation
```bash
# Via Claude Marketplace
claude plugin install ui-ux-pro-max

# Via CLI global (16+ assistants IA)
npx uipro-cli init --ai claude
```

## Commande de recherche
```bash
python3 src/ui-ux-pro-max/scripts/search.py "<query>" --domain <domain> [-n <max_results>]
```

## Domaines de recherche

| Domain | Description |
|--------|-------------|
| `product` | Recommandations par type de produit (SaaS, e-commerce, portfolio) |
| `style` | Styles UI + prompts AI et mots-clés CSS |
| `typography` | Pairings de polices avec imports Google Fonts |
| `color` | Palettes de couleurs par type de produit |
| `landing` | Structure de page et stratégies CTA |
| `chart` | Types de graphiques et recommandations de bibliothèques |
| `ux` | Bonnes pratiques et anti-patterns |

## Recherche par stack
```bash
python3 src/ui-ux-pro-max/scripts/search.py "<query>" --stack <stack>
```

### Stacks disponibles
`html-tailwind` (défaut), `react`, `nextjs`, `astro`, `vue`, `nuxtjs`, `nuxt-ui`, `svelte`, `swiftui`, `react-native`, `flutter`, `shadcn`, `jetpack-compose`

## Génération de design system complet
```bash
python3 src/ui-ux-pro-max/scripts/search.py "<query>" --design-system
```

---

## 10 Catégories prioritaires

### 1. Accessibilité (CRITIQUE)
- Ratios de contraste (4.5:1 minimum)
- États de focus visibles
- Navigation clavier
- Labels ARIA

### 2. Touch & Interaction (CRITIQUE)
- Cibles minimum 44×44px
- Espacement 8px minimum
- Feedback de chargement

### 3. Performance (HAUTE)
- Optimisation des images
- Lazy loading
- Prévention des layout shifts

### 4. Sélection de style (HAUTE)
- Cohérence avec le type de produit
- Icônes SVG
- Adaptation par plateforme

### 5. Layout & Responsive (HAUTE)
- Mobile-first
- Configuration viewport
- Pas de scroll horizontal

### 6. Typographie & Couleur (MOYENNE)
- Line height 1.5–1.75
- Tokens sémantiques
- Contraste 4.5:1

### 7. Animation (MOYENNE)
- Durée 150–300ms
- Transitions significatives
- Support reduced-motion

### 8. Formulaires & Feedback (MOYENNE)
- Labels visibles
- Placement des erreurs
- Progressive disclosure

### 9. Patterns de navigation (HAUTE)
- Comportement retour prévisible
- Deep linking
- Limites nav inférieure

### 10. Graphiques & Données (BASSE)
- Paires de couleurs accessibles
- Légendes et tooltips
- Simplification responsive

---

## Workflow d'implémentation

**Étape 1 :** Analyser les exigences (type de produit, audience, style, stack)

**Étape 2 :** Générer le design system complet avec `--design-system`

**Étape 3 :** Compléter avec des recherches de domaine pour les besoins spécifiques

**Étape 4 :** Appliquer les guidelines spécifiques au stack

---

## Ressources incluses
- **67 Styles UI** : glassmorphisme, brutalisme, minimalisme, AI-native, spatial computing...
- **161 Règles de raisonnement** par industrie (tech, finance, santé, e-commerce...)
- **161 Palettes de couleurs** alignées par type de produit
- **57 Pairings de polices** Google Fonts
- **99 Guidelines UX** : bonnes pratiques et accessibilité
- **25 Types de graphiques** avec recommandations de bibliothèques

---

## Checklist de vérification pré-livraison
- [ ] Qualité visuelle
- [ ] Réactivité aux interactions
- [ ] Cohérence du thème
- [ ] Correction du layout
- [ ] Conformité accessibilité

## Prérequis
Python 3.x (aucune dépendance externe requise)
