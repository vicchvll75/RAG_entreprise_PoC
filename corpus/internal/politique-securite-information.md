---
id: INT-003
title: Politique de sécurité de l'information
department: securite
classification: internal
allowed_roles: [employee, engineering, hr, finance, executive]
owner: RSSI (Responsable de la sécurité des systèmes d'information)
updated: 2026-03-01
---

# Politique de sécurité de l'information

## Classification des documents

Tout document produit chez Altiscope porte l'un des quatre niveaux suivants :

- **Public** : diffusable à l'extérieur (site web, plaquettes, communiqués).
- **Interne** : accessible à tous les collaborateurs, interdit à l'extérieur.
- **Confidentiel** : réservé à un département et à la direction générale.
- **Restreint** : réservé à une liste nominative ou à un comité (rémunérations individuelles,
  opérations de fusion-acquisition, trésorerie détaillée).

Les outils internes, y compris l'assistant documentaire basé sur l'IA, doivent appliquer
ces niveaux : un collaborateur ne doit jamais obtenir, même résumée ou paraphrasée, une
information d'un niveau auquel il n'a pas accès.

## Mots de passe et authentification

- Authentification multifacteur (MFA) obligatoire sur tous les outils SaaS.
- Mots de passe de 14 caractères minimum, gérés dans le coffre-fort 1Password d'entreprise.
- Renouvellement imposé uniquement en cas de suspicion de compromission.

## Données personnelles

Les données personnelles (clients, collaborateurs) sont traitées conformément au RGPD.
Il est interdit de coller des données personnelles ou des données clients dans un outil d'IA
générative externe non validé par le RSSI. Le délégué à la protection des données (DPO) est
joignable à dpo@altiscope.example.

## Signalement d'un incident de sécurité

Tout incident (phishing, perte d'un ordinateur, fuite de données) doit être signalé **dans
l'heure** au SOC via security@altiscope.example ou le canal #security-alerts. Si l'incident
touche la plateforme Sentinel en production, la procédure d'incident de production
(document ENG-002) s'applique en parallèle. En cas de violation de données personnelles,
le DPO notifie la CNIL sous 72 heures.

## Sanctions

Le non-respect de cette politique peut entraîner des sanctions disciplinaires selon la
procédure disciplinaire en vigueur.
