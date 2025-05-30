import os
import csv
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION GEMINI ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
configure(api_key=gemini_api_key)
gemini = GenerativeModel("models/gemini-2.0-flash")

# --- CHARGEMENT VECTORS WEB ---
web_vectorstore = Chroma(
    persist_directory="supdevinci-chatbot/chroma_site",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
web_retriever = web_vectorstore.as_retriever()

# --- CHARGEMENT VECTORS DOC ---
doc_vectorstore = Chroma(
    persist_directory="supdevinci-chatbot/chroma_reglement",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
doc_retriever = doc_vectorstore.as_retriever()

# --- EMOJIS POUR INTENTIONS ---
EMOJI_INTENT = {
    "web": "🌐",
    "doc": "📄",
    "action": "📝",
    "none": "❓"
}

# --- FONCTIONS D’AGENTS & ROUTEUR ---
def detect_intention(user_input):
    prompt = f"""
Tu joues le rôle d’un classificateur intelligent d’intention utilisateur.

Analyse attentivement le message suivant et classe-le uniquement dans **l’une des catégories suivantes** :

- **web** : si la question concerne les informations disponibles sur le site de SupdeVinci (ex : formations, campus, calendrier, admissions, contacts, etc.).
- **doc** : si la question concerne des documents internes comme le règlement intérieur, les politiques de l’école, les procédures, etc.
- **action** : si l’utilisateur cherche à effectuer une action concrète (inscription, dépôt de dossier, demande administrative, formulaire, etc.).
- **none** : si le message ne correspond à aucune de ces catégories ou est trop vague.

Donne uniquement l’étiquette correspondante (web, doc, action ou none), sans justification.

Message utilisateur : \"{user_input}\"
Réponse attendue (web, doc, action, none) :
"""
    response = gemini.generate_content(prompt)
    return response.text.strip().lower()

def handle_web_agent(user_input):
    docs = web_retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    En te basant sur les informations suivantes :
    {context}

    Réponds à la question suivante de manière claire et naturelle :
    {user_input}
    """
    response = gemini.generate_content(prompt)
    return response.text.strip()

def handle_doc_agent(user_input):
    docs = doc_retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    En te basant sur les documents internes suivants :
    {context}

    Réponds précisément à la question suivante :
    {user_input}
    """
    response = gemini.generate_content(prompt)
    return response.text.strip()

def handle_action_agent(user_input):
    prompt = f"""
Tu joues le rôle d’un assistant qui résume les informations clés d’une demande utilisateur.

Extrait :
- le prénom (s’il est mentionné)
- le nom (s’il est mentionné)
- l’action ou la demande principale de l’utilisateur
- la date actuelle (au format JJ/MM/AAAA)

Formate la réponse au format suivant :

Prénom : ...
Nom : ...
Action demandée : ...
Date : ...

Message utilisateur : "{user_input}"
    """
    response = gemini.generate_content(prompt).text.strip()

    # --- Sauvegarde CSV ---
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "demande": user_input,
        "résumé": response.replace("\n", " | ")
    }

    csv_path = "data/demandes.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    return f"Votre demande a bien été enregistrée.\n\nRésumé :\n{response}"

def main_router(user_input):
    intent = detect_intention(user_input)
    if intent == "web":
        return handle_web_agent(user_input), intent
    elif intent == "doc":
        return handle_doc_agent(user_input), intent
    elif intent == "action":
        return handle_action_agent(user_input), intent
    else:
        return "Désolé, je n'ai pas compris votre demande. Pouvez-vous reformuler ?", "none"

def clear_history():
    st.session_state.history = []

# --- CONFIGURATION UI ---
st.set_page_config(page_title="🤖 Assistant SupdeVinci")
st.title("Assistant intelligent SupdeVinci")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔧 Options")
    if st.button("🗑️ Supprimer l'historique"):
        clear_history()
        st.success("Historique supprimé !")

# --- SESSION STATE INIT ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- INPUT VIA st.chat_input ---
user_input = st.chat_input("Posez votre question sur SupdeVinci...")

if user_input:
    with st.spinner("Analyse en cours..."):
        answer, intent = main_router(user_input)

    # Ajout à l'historique
    st.session_state.history.append({
        "question": user_input,
        "answer": answer,
        "intent": intent
    })

# --- AFFICHAGE DE L'HISTORIQUE ---
for entry in reversed(st.session_state.history):
    st.markdown(
        f"""
        <div style="
            margin-bottom:15px;
            padding:10px;
            border-radius:8px;
            background:#e6f0ff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <b>Vous :</b> {entry['question']}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="
            margin-bottom:30px;
            padding:12px;
            border-radius:8px;
            background:#f0f4f8;
            border: 1px solid #d1d9e6;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
        ">
            <b>{EMOJI_INTENT.get(entry['intent'], '❓')} Assistant :</b> {entry['answer']}
        </div>
        """,
        unsafe_allow_html=True
    )
