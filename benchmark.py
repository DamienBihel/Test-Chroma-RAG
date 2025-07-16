import chromadb
import time
import psutil
import os
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import argparse

# Classe principale pour le benchmark de ChromaDB
class ChromaDBBenchmark:
    def __init__(self):
        # Liste de documents à indexer (mélange nature/techno)
        self.documents = [
            "Le ciel est bleu et dégagé aujourd'hui",
            "Le soleil brille de mille feux dorés",
            "La mer scintille sous les rayons du soleil",
            "Les oiseaux chantent dans les arbres verts",
            "Le vent souffle doucement dans les feuilles",
            "Les fleurs embaument le jardin printanier",
            "La montagne se dresse majestueuse",
            "Le lac reflète les nuages blancs",
            "La forêt regorge de vie sauvage",
            "L'herbe pousse dans la prairie verdoyante",
            "La technologie moderne transforme notre société",
            "L'intelligence artificielle révolutionne l'informatique",
            "Les algorithmes d'apprentissage automatique sont puissants",
            "Le machine learning analyse des données complexes",
            "Les réseaux de neurones simulent le cerveau humain",
            "La programmation Python est très populaire",
            "Les bases de données stockent des informations",
            "Le cloud computing offre des services distants",
            "L'analyse de données révèle des tendances cachées",
            "Les systèmes distribués gèrent la charge"
        ]
        # Requêtes de recherche à tester
        self.queries = [
            "nature et paysage",
            "technologie et informatique",
            "ciel et météo",
            "intelligence artificielle"
        ]
        # Dictionnaire pour stocker les fonctions d'embedding
        self.embedding_functions = {}
        # Dictionnaire pour stocker les résultats du benchmark
        self.results = {}

    def setup_embedding_functions(self):
        """Configure les différentes fonctions d'embedding à tester"""
        print("Configuration des fonctions d'embedding...")
        # Embedding par défaut de ChromaDB (souvent TF-IDF ou similaire)
        self.embedding_functions['default'] = None
        # SBERT MiniLM (rapide, sémantique)
        self.embedding_functions['sbert_mini'] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # SBERT Base (plus lourd, potentiellement plus précis)
        self.embedding_functions['sbert_base'] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        # OpenAI embedding (optionnel, nécessite une clé API)
        try:
            self.embedding_functions['openai'] = embedding_functions.OpenAIEmbeddingFunction(
                api_key="dummy_key"  # Remplacer par une vraie clé API si besoin
            )
        except:
            print("OpenAI embedding non disponible (clé API manquante)")

    def measure_memory_usage(self):
        """Mesure l'utilisation mémoire actuelle du processus (en MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def benchmark_embedding_function(self, name, embedding_func):
        """Benchmark une fonction d'embedding spécifique"""
        print(f"\n=== Benchmark: {name} ===")
        # Mesure mémoire avant l'opération
        memory_before = self.measure_memory_usage()
        # Création du client ChromaDB
        client = chromadb.Client()
        # Suppression de la collection si elle existe déjà
        try:
            client.delete_collection(name=f"benchmark_{name}")
        except:
            pass
        start_time = time.time()
        # Création de la collection avec ou sans embedding personnalisé
        if embedding_func:
            collection = client.create_collection(
                name=f"benchmark_{name}",
                embedding_function=embedding_func
            )
        else:
            collection = client.create_collection(name=f"benchmark_{name}")
        # Ajout des documents et mesure du temps
        add_start = time.time()
        collection.add(
            ids=[f"doc_{i}" for i in range(len(self.documents))],
            documents=self.documents
        )
        add_time = time.time() - add_start
        # Mesure mémoire après ajout
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before
        # Mesure du temps de requête pour chaque query
        query_times = []
        query_results = []
        for query in self.queries:
            query_start = time.time()
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            query_time = time.time() - query_start
            query_times.append(query_time)
            query_results.append(results)
        total_time = time.time() - start_time
        avg_query_time = sum(query_times) / len(query_times)
        # Stockage des résultats pour ce modèle
        self.results[name] = {
            'total_time': total_time,
            'add_time': add_time,
            'avg_query_time': avg_query_time,
            'memory_usage': memory_usage,
            'query_results': query_results
        }
        # Affichage des métriques principales
        print(f"Temps total: {total_time:.3f}s")
        print(f"Temps d'ajout: {add_time:.3f}s")
        print(f"Temps de requête moyen: {avg_query_time:.3f}s")
        print(f"Utilisation mémoire: {memory_usage:.1f}MB")
        # Exemple de résultats pour la première requête
        print(f"\nExemple de résultats pour '{self.queries[0]}':")
        for i, doc in enumerate(query_results[0]['documents'][0]):
            distance = query_results[0]['distances'][0][i]
            print(f"  {i+1}. {doc[:50]}... (distance: {distance:.3f})")

    def run_benchmark(self):
        """Exécute le benchmark complet sur tous les embeddings"""
        print("=== BENCHMARK CHROMADB AVEC DIFFÉRENTS EMBEDDINGS ===")
        print(f"Nombre de documents: {len(self.documents)}")
        print(f"Nombre de requêtes: {len(self.queries)}")
        self.setup_embedding_functions()
        # Boucle sur chaque embedding à tester
        for name, func in self.embedding_functions.items():
            if name == 'openai' and func is None:
                continue  # Ignore si OpenAI non dispo
            try:
                self.benchmark_embedding_function(name, func)
            except Exception as e:
                print(f"Erreur avec {name}: {e}")
        self.display_comparison()

    def display_comparison(self):
        """Affiche un tableau comparatif des résultats du benchmark"""
        print("\n" + "="*80)
        print("COMPARAISON DES PERFORMANCES")
        print("="*80)
        # En-têtes du tableau
        print(f"{'Modèle':<15} {'Temps total':<12} {'Ajout':<8} {'Requête':<9} {'Mémoire':<10}")
        print("-" * 80)
        # Affichage des résultats pour chaque embedding
        for name, results in self.results.items():
            print(f"{name:<15} {results['total_time']:<12.3f} {results['add_time']:<8.3f} "
                  f"{results['avg_query_time']:<9.3f} {results['memory_usage']:<10.1f}")
        # Recommandations automatiques
        print("\n" + "="*80)
        print("RECOMMANDATIONS")
        print("="*80)
        if 'default' in self.results and 'sbert_mini' in self.results:
            default_time = self.results['default']['total_time']
            sbert_time = self.results['sbert_mini']['total_time']
            if sbert_time < default_time * 1.2:  # SBERT pas trop lent
                print("✅ SBERT recommandé pour une meilleure qualité sémantique")
            else:
                print("⚡ Embedding par défaut recommandé pour la vitesse")
        # Affichage du plus rapide et du moins gourmand en mémoire
        fastest = min(self.results.items(), key=lambda x: x[1]['total_time'])
        print(f"🏃 Plus rapide: {fastest[0]} ({fastest[1]['total_time']:.3f}s)")
        lowest_memory = min(self.results.items(), key=lambda x: x[1]['memory_usage'])
        print(f"💾 Moins de mémoire: {lowest_memory[0]} ({lowest_memory[1]['memory_usage']:.1f}MB)")

# Fonction principale pour lancer le benchmark via la ligne de commande

def main():
    parser = argparse.ArgumentParser(description='Benchmark ChromaDB avec différents embeddings')
    parser.add_argument('--save', action='store_true', help='Sauvegarder les résultats')
    args = parser.parse_args()
    benchmark = ChromaDBBenchmark()
    benchmark.run_benchmark()
    # Option de sauvegarde des résultats dans un fichier texte
    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write("=== RÉSULTATS DU BENCHMARK CHROMADB ===\n\n")
            for name, results in benchmark.results.items():
                f.write(f"{name}:\n")
                f.write(f"  Temps total: {results['total_time']:.3f}s\n")
                f.write(f"  Temps d'ajout: {results['add_time']:.3f}s\n")
                f.write(f"  Temps de requête moyen: {results['avg_query_time']:.3f}s\n")
                f.write(f"  Utilisation mémoire: {results['memory_usage']:.1f}MB\n\n")
        print(f"\nRésultats sauvegardés dans {filename}")

# Point d'entrée du script
if __name__ == "__main__":
    main()