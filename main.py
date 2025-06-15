import os
import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from ollama import Client

# --- CONFIGURATION ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legalai"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
LLM_MODEL = 'HuggingFaceH4/zephyr-7b-beta' # A good starting open-source model
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-small3.1:24b")  # Default fallback model

# --- INITIALIZATION ---
def initialize_services():
    """Initializes and returns all required service clients."""
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Initialize Embedding Model
    # This model is great for multilingual semantic search
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize LLM from Ollama
    llm_pipeline = Client(host="http://localhost:11434")
    
    return pc, embedding_model, llm_pipeline

# --- DATA & PINECONE MANAGEMENT ---
def load_and_prepare_data(filepath="data/dummy_moroccan_law.json"):
    """Loads legal articles from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_pinecone_index_if_not_exists(pc: Pinecone):
    """Checks if the Pinecone index exists, and creates it if it doesn't."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768, # Dimension for the chosen embedding model
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

def embed_and_upsert(index, data, model):
    """Embeds legal data and upserts it into the Pinecone index."""
    print("Embedding and upserting data...")
    # Prepare vectors for upsert
    vectors_to_upsert = []
    for item in data:
        # Create a combined text for a richer embedding
        combined_text = f"Article {item['article']}: {item['title']}. Content: {item['content']}"
        embedding = model.encode(combined_text).tolist()
        vector = {
            "id": item['article'],
            "values": embedding,
            "metadata": {"title": item['title'], "content": item['content']}
        }
        vectors_to_upsert.append(vector)
    
    # Upsert in batches for efficiency
    index.upsert(vectors=vectors_to_upsert, batch_size=100)
    print(f"Upserted {len(vectors_to_upsert)} vectors.")

# --- RETRIEVAL-AUGMENTED GENERATION (RAG) ---
def retrieve_context(query, index, model, top_k=3):
    """Retrieves relevant legal context from Pinecone."""
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Format the retrieved data into a string context
    context = "CONTEXTE JURIDIQUE PERTINENT (Source: Code Marocain des Obligations et des Contrats):\n"
    for match in results['matches']:
        context += f"- Article {match['id']} ({match['metadata']['title']}): {match['metadata']['content']}\n"
    return context

def build_prompt(user_request, context, task_type="generate"):
    """Builds a detailed, grounded prompt for the LLM."""
    if task_type == "generate":
        system_message = """
        Vous êtes un assistant juridique spécialisé dans le droit marocain.
        Votre tâche est de générer des clauses de contrat ou des contrats complets.
        La génération doit etre en français, il sera bien de reflechir étape par étape (aussi en français si possible).
        Vous devez utiliser le contexte juridique fourni pour vous assurer que la clause est conforme aux lois marocaines.
        Vous devez également vous assurer que la clause répond aux besoins de l'entreprise.
        En ce qui concerne le format du contrat, vous devez suivre les conventions de rédaction des contrats au Maroc (Articles, Clauses, etc.).
        Vous DEVEZ vous baser EXCLUSIVEMENT sur le contexte juridique fourni.
        NE PAS inventer d'informations. Citez les articles de loi pertinents lorsque c'est possible.
        Soyez précis, formel et clair.
        """
        human_message = f"""
        {context}
        ---
        DEMANDE DE L'UTILISATEUR:
        Rédigez une clause ou un contrat basé sur la demande suivante : "{user_request}"
        ---
        RÉPONSE (en français):
        """
    else: # Evaluate task
        system_message = """
        Vous êtes un analyste de risques juridiques spécialisé dans le droit marocain.
        Votre tâche est d'évaluer une clause de contrat en fonction des lois marocaines fournies et des objectifs de l'entreprise.
        Identifiez les risques, les ambiguïtés ou les non-conformités.
        Fondez TOUTE votre analyse sur le contexte juridique fourni.
        """
        human_message = f"""
        {context}
        ---
        OBJECTIFS DE L'ENTREPRISE:
        {user_request['goals']}
        ---
        CLAUSE DU CONTRAT À ÉVALUER:
        "{user_request['clause']}"
        ---
        ANALYSE DES RISQUES (en français):
        """

    # Using the Zephyr chat template
    prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{human_message}</s>\n<|assistant|>"
    return prompt

def get_llm_response(llm_pipeline: Client, prompt: str, model: str = None):
    """Gets a response from the LLM using the specified model."""
    if model is None:
        model = OLLAMA_MODEL
        if model is None:
            raise ValueError("No model specified and OLLAMA_MODEL environment variable is not set")
    
    print(f"Generating LLM response with model: {model}")
    sequences = llm_pipeline.generate(
        model=model,
        prompt=prompt,
    )
    return sequences['response']