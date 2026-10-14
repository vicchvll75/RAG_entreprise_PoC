---
id: ENG-002
title: Procédure d'incident de production
department: engineering
classification: confidential
allowed_roles: [engineering, executive]
owner: Head of SRE
updated: 2026-03-18
---

# Procédure d'incident de production

## Niveaux de sévérité

| Niveau | Définition | Délai de prise en charge |
|---|---|---|
| SEV1 | Plateforme indisponible ou perte de données pour plusieurs clients | 15 minutes |
| SEV2 | Fonctionnalité majeure dégradée (alertes en retard > 5 min) | 30 minutes |
| SEV3 | Problème limité à un client ou contournement disponible | 4 heures ouvrées |
| SEV4 | Anomalie mineure, sans impact client | Prochain sprint |

## Astreinte

Une rotation d'astreinte hebdomadaire couvre 24 h/24 et 7 j/7 avec un ingénieur principal et
un ingénieur secondaire. L'astreinte est rémunérée par une prime forfaitaire de **350 € par
semaine**, plus une récupération en temps pour toute intervention de nuit. L'outil de paging
est PagerDuty.

## Déroulé d'un incident SEV1 ou SEV2

1. L'ingénieur d'astreinte accuse réception dans PagerDuty et ouvre le canal Slack
   `#inc-AAAAMMJJ-titre`.
2. Il nomme un **Incident Commander** (IC) et un responsable communication.
3. La page de statut publique est mise à jour toutes les 30 minutes.
4. Pour un SEV1, le CTO et le directeur du support client sont prévenus immédiatement.
5. Si un incident de sécurité est suspecté, le SOC est prévenu selon la politique de sécurité
   de l'information (INT-003).

## Post-mortem

Tout SEV1 ou SEV2 donne lieu à un post-mortem sans recherche de coupable, rédigé sous
5 jours ouvrés, avec actions correctives suivies dans Jira. Les crédits de service dus aux
clients en cas de violation du SLA de 99,9 % sont calculés par la Direction financière.

## Gel des déploiements

Pendant un incident SEV1 ou SEV2 ouvert, tout déploiement en production est gelé, sauf
correctif validé par l'Incident Commander (voir les standards de déploiement, ENG-003).
