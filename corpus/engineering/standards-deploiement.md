---
id: ENG-003
title: Standards de déploiement
department: engineering
classification: confidential
allowed_roles: [engineering, executive]
owner: Head of Platform
updated: 2026-02-27
---

# Standards de déploiement

## Branches et revue

- Développement sur des branches courtes, fusionnées dans `main` via pull request.
- **Deux approbations** requises pour tout service critique (`predict-api`, `alert-engine`,
  `ingest-gateway`), une seule pour les autres.
- La CI (GitHub Actions) exécute tests unitaires, tests d'intégration, analyse SAST et scan
  des dépendances. Aucune fusion n'est possible si la CI échoue.

## Environnements

1. **dev** : déploiement automatique à chaque fusion.
2. **staging** : copie anonymisée de la production, déploiement quotidien.
3. **production** : déploiement progressif (canary) via Argo Rollouts.

## Déploiement canary

Le trafic est basculé par paliers : 5 %, 25 %, 50 %, puis 100 %, avec 15 minutes d'observation
entre chaque palier. Le rollback est automatique si le taux d'erreur dépasse 1 % ou si la
latence p95 de `predict-api` dépasse 150 ms.

## Fenêtres de déploiement

- Déploiements en production autorisés du lundi au jeudi, de 9 h à 16 h (heure de Paris).
- **Aucun déploiement le vendredi**, ni la veille d'un jour férié, sauf correctif urgent.
- Gel complet des déploiements pendant un incident SEV1/SEV2 (voir ENG-002) et pendant la
  semaine de clôture comptable annuelle demandée par la Direction financière.

## Modèles de machine learning

Un nouveau modèle Sentinel Predict n'est promu en production que si son rappel sur les pannes
est au moins égal à celui du modèle courant sur le jeu de validation, et après un test en
shadow mode de 7 jours.
