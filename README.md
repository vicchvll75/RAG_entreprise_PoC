# RAG Enterprise — assistant documentaire avec RBAC, garde-fous et évaluation

Assistant de questions-réponses sur la documentation interne d'une entreprise fictive
(Altiscope, 15 documents de « public » à « restreint »). Chaque collaborateur n'obtient de
réponse qu'à partir des documents que son rôle l'autorise à lire.

## Architecture

```
question + rôle
   │
   ▼
Recherche hybride  ── pré-filtre RBAC (Chroma `where` + index BM25 filtré)
 dense (mistral-embed) + BM25 → fusion RRF → top-k chunks
   │
   ▼
Génération (Mistral, température 0) — réponse uniquement depuis le contexte, citations [DOC-ID],
phrase de refus fixe si l'information manque
   │
   ▼
Garde-fous de sortie
 • code canari d'un document interdit dans la réponse → bloqué
 • citation d'un document non remonté → bloqué (signe d'hallucination)
 • (option) fidélité Ragas < seuil → bloqué
```

Le RBAC est appliqué **avant** la recherche : un chunk interdit n'atteint jamais le LLM, donc
il ne peut pas fuiter, même paraphrasé. Filtrer après coup, ou demander au LLM de « ne pas
divulguer », ne suffit pas.

## Démarrage

```bash
pip install -r requirements.txt
cp .env.example .env            # renseigner MISTRAL_API_KEY
python scripts/ingest.py        # construit l'index Chroma
uvicorn rag.api:app --app-dir src
curl -X POST localhost:8000/ask -H "X-User-Role: employee" \
     -H "Content-Type: application/json" -d '{"question": "Combien de jours de télétravail ?"}'
```

## Évaluation

```bash
python eval/run_eval.py --retrieval-only   # sans LLM : gratuit, à chaque commit
python eval/run_eval.py --no-ragas         # + génération, refus, fuites
python eval/run_eval.py                    # + Ragas (juge LLM)
pytest                                      # tests unitaires, RBAC et garde-fous
```

Le jeu `eval/golden_set.jsonl` contient 40 questions annotées :

| type | n | comportement attendu |
|---|---|---|
| `answerable` | 27 | répondre, avec les bons documents (dont 4 questions multi-documents) |
| `forbidden` | 8 | refuser : l'information existe mais le rôle n'y a pas accès (dont 1 injection de prompt) |
| `unanswerable` | 5 | refuser : l'information n'existe pas dans le corpus (pièges à hallucination) |

Le script vérifie aussi le jeu lui-même : une question « répondable » qui pointerait vers un
document interdit au rôle fait échouer le chargement.

### Les métriques et ce qu'elles mesurent

**1. Retrieval — déterministe, sans LLM** (au niveau document, top-k dédoublonné)

- **Recall@k** : part des documents pertinents retrouvés. La métrique prioritaire : si le bon
  document n'est pas remonté, le LLM ne peut pas bien répondre.
- **Precision@k** : part des documents remontés qui sont pertinents. Mécaniquement basse
  quand une seule source est pertinente et k = 5 ; c'est le compromis rappel / bruit.
- **F1@k** : moyenne harmonique des deux.
- **MRR** : rang du premier document pertinent (1 = en tête).

**2. Refus — classification binaire, classe positive = « doit refuser »**

- **Précision** basse → le système refuse à tort (trop prudent, inutile).
- **Rappel** bas → il répond alors qu'il devrait refuser (hallucination ou fuite).
- **F1** résume les deux ; la matrice de confusion complète est dans le rapport.

**3. Sécurité — tolérance zéro**

- `rbac.retrieval_violations` : chunks interdits remontés (doit être 0).
- `rbac.leaked_answers` : réponses contenant une chaîne interdite (montant, code de suivi).
  Les documents sensibles contiennent un **code canari** unique : sa présence dans une réponse
  prouve une fuite, même si le reste est paraphrasé.

**4. Ragas — juge LLM**

| métrique | question posée | rôle |
|---|---|---|
| `faithfulness` | Chaque affirmation de la réponse est-elle soutenue par le contexte ? | **anti-hallucination** |
| `answer_relevancy` | La réponse traite-t-elle la question posée ? | hors-sujet |
| `context_precision` | Les chunks utiles sont-ils classés en tête ? | qualité du classement |
| `context_recall` | Le contexte contient-il tout ce qu'exige la réponse de référence ? | complétude du retrieval |
| `factual_correctness_f1` | Précision / rappel / F1 des affirmations vs la référence | justesse de la réponse |

`faithfulness` et `answer_relevancy` ne sont calculées que sur les réponses effectivement
données ; un refus à tort est compté dans le F1 du refus et obtient 0 en `factual_correctness`.

### Quality gate

`run_eval.py` renvoie le code 1 si un seuil n'est pas tenu (`THRESHOLDS` dans le script), ce
qui permet de bloquer une PR en CI : recall@k ≥ 0,90, MRR ≥ 0,80, F1 refus ≥ 0,85,
fidélité ≥ 0,85, aucune fuite. Chaque exécution écrit `eval/results/<horodatage>.json` et `.md`
avec le détail par question, la latence p50/p95 et les tokens consommés.

## Limites connues

- Le juge LLM a ses propres biais : il faut vérifier un échantillon de scores à la main et
  utiliser un modèle juge plus fort que le modèle de génération.
- 40 questions donnent un ordre de grandeur, pas un intervalle de confiance serré.
- Le rôle passe dans un header ; en production il viendrait d'un JWT validé par le SSO.
- BM25 est recalculé en mémoire au démarrage : adapté à ce corpus, pas à des millions de
  chunks (on passerait aux vecteurs creux de Qdrant ou à OpenSearch).
