import streamlit as st
import os
import json
from main import (
    initialize_services,
    load_and_prepare_data,
    create_pinecone_index_if_not_exists,
    embed_and_upsert,
    retrieve_context,
    build_prompt,
    get_llm_response
)
from ollama import Client

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Générateur de Contrats Marocains (PoC)",
    page_icon="⚖️",
    layout="wide"
)

# --- STATE MANAGEMENT ---
# Using session state to cache resources and avoid reloading
if 'services_initialized' not in st.session_state:
    st.session_state.services_initialized = False
if 'data_indexed' not in st.session_state:
    st.session_state.data_indexed = False

# --- HELPER FUNCTIONS ---
@st.cache_resource
def setup_services():
    """Cached function to initialize services once."""
    return initialize_services()

def get_available_ollama_models():
    """Fetches the list of available models from Ollama."""
    try:
        client = Client(host="http://localhost:11434")
        response = client.list()
        model_names = [model.model for model in response.models]
        return model_names
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")
        return []

# --- MAIN APP ---
st.title("⚖️ Assistant Juridique Marocain (Preuve de Concept)")

# --- LEGAL DISCLAIMER ---
st.warning(
    "**AVERTISSEMENT LÉGAL IMPORTANT:**\n"
    "Cet outil est une preuve de concept et ne remplace en aucun cas les conseils d'un avocat qualifié. "
    "Toutes les informations et les documents générés DOIVENT être examinés par un professionnel du droit avant toute utilisation."
)

# --- SETUP & INITIALIZATION ---
st.sidebar.header("Configuration")
st.session_state.pinecone_api_key = st.sidebar.text_input("Clé API Pinecone", type="password")

# Get available Ollama models
available_models = get_available_ollama_models()

if not available_models:
    st.sidebar.error("Aucun modèle Ollama disponible. Vérifiez que Ollama est en cours d'exécution.")
    st.stop()

# Model selection dropdown
selected_model = st.sidebar.selectbox(
    "Sélectionner le modèle LLM:",
    available_models,
    index=0 if available_models else None,
    help="Choisissez le modèle Ollama à utiliser pour la génération de texte"
)

# Display selected model info
st.sidebar.info(f"Modèle sélectionné: {selected_model}")

if st.sidebar.button("Initialiser les Services"):
    if st.session_state.pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_api_key
        os.environ["OLLAMA_MODEL"] = selected_model
        with st.spinner("Initialisation des services... Veuillez patienter."):
            try:
                st.session_state.pc, st.session_state.embedding_model, st.session_state.llm_pipeline = setup_services()
                st.session_state.services_initialized = True
                st.sidebar.success("Services initialisés !")
            except Exception as e:
                st.sidebar.error(f"Erreur d'initialisation: {e}")
    else:
        st.sidebar.warning("Veuillez entrer votre clé API Pinecone.")

if st.session_state.services_initialized:
    if st.sidebar.button("Indexer les Données Juridiques"):
        with st.spinner("Création de l'index et indexation des données..."):
            try:
                # 1. Create index if it doesn't exist
                create_pinecone_index_if_not_exists(st.session_state.pc)
                st.session_state.pinecone_index = st.session_state.pc.Index("legalai")
                
                # 2. Load data
                legal_data = load_and_prepare_data()
                
                # 3. Embed and upsert data
                embed_and_upsert(st.session_state.pinecone_index, legal_data, st.session_state.embedding_model)
                st.session_state.data_indexed = True
                st.sidebar.success("Données indexées avec succès !")
            except Exception as e:
                st.sidebar.error(f"Erreur d'indexation: {e}")
else:
    st.info("Veuillez initialiser les services via la barre latérale pour commencer.")


# --- APPLICATION TABS ---
if st.session_state.services_initialized and st.session_state.data_indexed:
    tab1, tab2 = st.tabs(["Générateur de Contrat", "Évaluateur de Contrat"])

    with tab1:
        st.header("Générateur de Contrat")
        st.markdown("Décrivez le contrat ou la clause que vous souhaitez générer. Soyez aussi précis que possible.")
        
        user_request_gen = st.text_area(
            "Votre demande:", 
            height=150,
            placeholder="Exemple: 'Je veux une clause pour un contrat de prêt de 5000 MAD entre deux personnes, sans intérêts, remboursable en 10 mois.'"
        )

        if st.button("Générer le brouillon"):
            if user_request_gen:
                with st.spinner("Recherche des articles de loi pertinents..."):
                    context = retrieve_context(user_request_gen, st.session_state.pinecone_index, st.session_state.embedding_model)
                
                st.info("Contexte Juridique Retrouvé:", icon="📚")
                st.text(context)

                with st.spinner("Génération du brouillon par l'IA..."):
                    prompt = build_prompt(user_request_gen, context, task_type="generate")
                    response = get_llm_response(st.session_state.llm_pipeline, prompt, model=selected_model)

                st.success("Brouillon Généré:", icon="📄")
                st.markdown(response)
            else:
                st.error("Veuillez entrer une description de votre besoin.")

    with tab2:
        st.header("Évaluateur de Risques")
        st.markdown("Collez une clause de contrat et décrivez vos objectifs pour recevoir une analyse de risques.")

        clause_to_evaluate = st.text_area(
            "Clause du contrat à évaluer:", 
            height=150, 
            placeholder="Exemple: 'L'emprunteur s'engage à rembourser la totalité de la somme due à une date qui sera convenue ultérieurement.'"
        )
        company_goals = st.text_area(
            "Vos objectifs/votre situation:", 
            height=100, 
            placeholder="Exemple: 'Je suis le prêteur. Mon objectif est d'assurer un remboursement rapide et de minimiser les ambiguïtés sur la date d'échéance.'"
        )

        if st.button("Évaluer les Risques"):
            if clause_to_evaluate and company_goals:
                # For evaluation, the query for retrieval should be the clause itself
                query = clause_to_evaluate + " " + company_goals
                with st.spinner("Recherche des articles de loi pertinents..."):
                    context = retrieve_context(query, st.session_state.pinecone_index, st.session_state.embedding_model)
                
                st.info("Contexte Juridique Retrouvé:", icon="📚")
                st.text(context)

                with st.spinner("Analyse des risques par l'IA..."):
                    eval_request = {"clause": clause_to_evaluate, "goals": company_goals}
                    prompt = build_prompt(eval_request, context, task_type="evaluate")
                    response = get_llm_response(st.session_state.llm_pipeline, prompt, model=selected_model)

                st.success("Analyse de Risques:", icon="🛡️")
                st.markdown(response)
            else:
                st.error("Veuillez remplir tous les champs.")
else:
    st.markdown("---")
    st.markdown("**Prochaines étapes:**")
    st.markdown("1. Entrez votre clé API Pinecone dans la barre latérale.")
    st.markdown("2. Cliquez sur 'Initialiser les Services'.")
    st.markdown("3. Une fois les services prêts, cliquez sur 'Indexer les Données Juridiques'.")