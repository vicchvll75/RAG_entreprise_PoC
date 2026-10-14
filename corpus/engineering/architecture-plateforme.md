---
id: ENG-001
title: Architecture de la plateforme Sentinel
department: engineering
classification: confidential
allowed_roles: [engineering, executive]
owner: CTO
updated: 2026-04-12
---

# Architecture de la plateforme Sentinel

## Vue d'ensemble

Sentinel est hébergée sur AWS, régions **eu-west-3 (Paris)** pour les clients européens et
ca-central-1 (Montréal) pour les clients nord-américains. Les données ne quittent jamais la
région du client.

## Chaîne de données

1. **Ingestion** : les passerelles IoT envoient les mesures en MQTT vers AWS IoT Core
   (environ 2,1 milliards de mesures par jour).
2. **Streaming** : les messages sont poussés dans Apache Kafka (cluster MSK, 12 brokers).
3. **Stockage chaud** : TimescaleDB conserve 90 jours de données brutes à la seconde.
4. **Stockage froid** : les données plus anciennes sont compactées en Parquet sur S3 et
   interrogées via Athena.
5. **Prédiction** : les modèles Sentinel Predict (gradient boosting et LSTM) sont entraînés
   chaque nuit sur SageMaker et servis par un service FastAPI sur EKS.
6. **Alerting** : un service d'alerte en Go évalue les prédictions et notifie les clients
   (e-mail, SMS, application Sentinel Field).

## Services principaux

- `ingest-gateway` (Go) : validation et normalisation des messages capteurs.
- `predict-api` (Python, FastAPI) : inférence, latence cible p95 < 150 ms.
- `alert-engine` (Go) : règles d'alerte et déduplication.
- `tenant-service` (Kotlin) : gestion des clients, sites et droits.

## Observabilité

Métriques dans Prometheus et Grafana, logs dans OpenSearch, traces OpenTelemetry. Les SLO
(disponibilité 99,9 %, latence d'alerte < 60 s) sont suivis sur un tableau de bord partagé avec
l'équipe d'astreinte. Toute violation de SLO déclenche la procédure d'incident de production
(ENG-002).

## Dette technique connue

Le `tenant-service` est un monolithe historique dont la découpe est prévue au S2 2026.
La migration de TimescaleDB vers une version managée est budgétée dans le budget 2026.
