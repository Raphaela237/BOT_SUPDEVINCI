import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader


# Dossiers des bases vectorielles
chroma_site_path = "supdevinci-chatbot/chroma_site"
chroma_reglement_path = "supdevinci-chatbot/chroma_reglement"

# Fichiers source JSON
site_json_path = "data/processed/site_chunks.json"
reglement_json_path = "data/processed/site_chunks.json" 

def charger_documents_json(path):
    loader = JSONLoader(file_path=path, jq_schema='.[]', text_content=True)
    return loader.load()

def creer_vectorstore(documents, vectorstore_path, embeddings):
    print(f"Création base vectorielle Chroma : {vectorstore_path}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    print(f"Base vectorielle sauvegardée dans : {vectorstore_path}")
    return vectorstore

def main():
    # Chargement local du modèle HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device':'cpu'}
)

    print("Chargement des documents du site...")
    site_docs = charger_documents_json(site_json_path)
    creer_vectorstore(site_docs, chroma_site_path, embeddings)

    print("Chargement des documents règlementaires...")
    reglement_docs = charger_documents_json(reglement_json_path)
    creer_vectorstore(reglement_docs, chroma_reglement_path, embeddings)

if __name__ == "__main__":
    main()
