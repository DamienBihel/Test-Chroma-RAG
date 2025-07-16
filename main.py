# Importation de la bibliothèque ChromaDB pour la gestion de bases de données vectorielles
import chromadb
from chromadb.utils import embedding_functions
# Importation de SentenceTransformer pour les embeddings SBERT
from sentence_transformers import SentenceTransformer

# Création d'un client ChromaDB (point d'entrée pour manipuler les collections)
chroma_client = chromadb.Client()

# Création d'une fonction d'embedding SBERT compatible avec ChromaDB
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Création d'une nouvelle collection nommée "test" avec fonction d'embedding SBERT
collection = chroma_client.create_collection(
    name="test",
    embedding_function=sentence_transformer_ef
)

# Ajout de documents à la collection
# Chaque document doit avoir un identifiant unique (ids) et un texte (documents)
collection.add(
    ids=["id1", "id2", "id3"],  # Liste des identifiants uniques
    documents=[
        "Le ciel est bleu", 
        "Le soleil est jaune",
        "Le ciel est une ode à l'espoir"
    ]  # Liste des textes à indexer
)

# Requête : on cherche les documents les plus proches du texte "Ciel"
results = collection.query(
    query_texts=["Ciel"],  # Texte de la requête (ce qu'on cherche)
    n_results=2            # Nombre de résultats souhaités
)

# Affichage lisible des résultats
# On affiche pour chaque résultat : l'id, le texte et la distance de similarité
print("Résultats de la recherche :")
for idx, doc in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"- id : {doc_id}\n  texte : {doc}\n  distance : {distance:.3f}\n")