# /gsd:debug

**But :** Débogage systématique via un sous-agent isolé avec 200k tokens de contexte frais.

## Usage
```
/gsd:debug [--diagnose]
```

## Flags
- `--diagnose` — Analyse des causes racines sans appliquer de corrections (retourne un rapport structuré)

## Processus
1. Initialiser contexte, vérifier sessions debug actives
2. Pour nouveaux problèmes, recueillir 5 détails de symptômes :
   - Comportement attendu
   - Comportement réel
   - Messages d'erreur
   - Chronologie
   - Étapes de reproduction
3. Spawn du sous-agent `gsd-debugger` avec les données de symptômes
4. Traitement du retour : cause racine trouvée / debug terminé / checkpoint / investigation non concluante
5. Pour continuations : nouveaux agents avec état précédent du fichier debug

## Principe
Méthode scientifique — phases d'investigation isolées qui préservent l'efficacité du contexte tout en maintenant la rigueur diagnostique.

## Marqueurs de succès
- Sessions actives vérifiées
- Symptômes recueillis
- Sous-agent spawné avec contexte approprié
- Checkpoints traités
- Cause racine vérifiée avant correction
