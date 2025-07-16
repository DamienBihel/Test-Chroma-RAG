# Test-Chroma-RAG

Ce projet propose une démonstration de l'utilisation de ChromaDB pour la recherche sémantique sur des documents en français, avec comparaison de différents modèles d'embedding.

## Prérequis

- Python 3.8+
- pip

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <url-du-repo>
   cd Test-Chroma-RAG
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Script principal : main.py
Ce script crée une collection ChromaDB, ajoute quelques documents, puis effectue une recherche sémantique simple.

Exécution :
```bash
python main.py
```

### 2. Benchmark : benchmark.py
Ce script compare les performances de différents modèles d'embedding (par défaut, SBERT MiniLM, SBERT Base, OpenAI si dispo) sur des requêtes variées.

Exécution simple :
```bash
python benchmark.py
```

Pour sauvegarder les résultats dans un fichier texte :
```bash
python benchmark.py --save
```

## Structure du projet
- `main.py` : exemple simple d'utilisation de ChromaDB
- `benchmark.py` : benchmark complet avec plusieurs embeddings
- `requirements.txt` : dépendances Python

## Notes
- Pour utiliser l'embedding OpenAI, renseignez une vraie clé API dans le code.
- Les modèles SBERT sont téléchargés automatiquement via `sentence-transformers`.

## Références
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/) 