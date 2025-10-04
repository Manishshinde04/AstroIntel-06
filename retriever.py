import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Knowledge Base ---
knowledge_base = [
    {"id": 1, "context": "E. coli and Bacillus subtilis are commonly studied microbes on the ISS in microgravity experiments to understand their behavior, growth, and mutation in space."},
    {"id": 2, "context": "Microgravity can surprisingly alter bacterial growth rates, increase their resistance to antibiotics, and change their gene expression, which is a key area of study for long-duration space missions."},
    {"id": 3, "context": "Leafy greens like lettuce and radishes, as well as dwarf wheat, have been successfully grown in microgravity. They are chosen for their fast growth cycles and nutritional value for astronauts."},
    {"id": 4, "context": "To combat bone density loss in a zero-gravity environment, astronauts must follow a strict regimen of resistance exercises using special equipment, maintain a calcium-rich diet, and sometimes use specific medications."},
    {"id": 5, "context": "Some incredibly resilient microbes called extremophiles, as well as certain bacterial spores, have been shown to survive for years when exposed to the vacuum and radiation of space, raising questions about the interplanetary transfer of life."},
    {"id": 6, "context": "Astrobiology is the scientific field dedicated to studying the origin, evolution, distribution, and future of life in the universe. It combines principles of biology, chemistry, and astronomy to explore the possibility of life beyond Earth."}
]

# --- Semantic Search Engine ---
class Context_Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing Context Retriever...")
        self.embedding_model = SentenceTransformer(model_name)
        self.contexts = [item['context'] for item in knowledge_base]
        self.embeddings = self._get_or_create_embeddings()
        print("Context Retriever ready.")

    def _get_or_create_embeddings(self):
        embedding_file = 'kb_embeddings.npy'
        if os.path.exists(embedding_file):
            print("Loading cached embeddings...")
            return np.load(embedding_file)
        else:
            print("Creating and caching new embeddings...")
            embeddings = self.embedding_model.encode(self.contexts, convert_to_tensor=False)
            np.save(embedding_file, embeddings)
            return embeddings

def retrieve_context(self, query: str, threshold=0.2): # Changed from 0.3
    query_embedding = self.embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, self.embeddings)[0]
    most_similar_index = np.argmax(similarities)
    if similarities[most_similar_index] < threshold:
        return None
    return self.contexts[most_similar_index]