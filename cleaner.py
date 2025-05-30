import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import os

def nettoyer_texte_site_web(raw_text):
    # Supprimer les lignes inutiles (formulaires, RGPD, champs)
    lignes = raw_text.splitlines()
    filtres = [
        r'^\*', r'RGPD', r'Adresse', r'Code postal', r'Nom', r'Téléphone',
        r'Email', r'Prénom', r'Poster une offre', r'En soumettant ce formulaire',
        r'Je suis intéressé par', r'Descriptif', r'job dating', r'animer des cours'
    ]
    lignes_utiles = []
    for ligne in lignes:
        ligne = ligne.strip()
        if ligne and not any(re.search(f, ligne, re.IGNORECASE) for f in filtres):
            lignes_utiles.append(ligne)

    # Recolle les fragments (ex : mots coupés entre lignes)
    texte = ' '.join(lignes_utiles)
    texte = re.sub(r'\s+', ' ', texte)  # Nettoie les espaces multiples
    return texte.strip()


def lire_et_nettoyer_pdfs(pdf_paths):
    texte_total = ''
    for path in pdf_paths:
        reader = PyPDF2.PdfReader(path)
        for page in reader.pages:
            texte = page.extract_text()
            texte_total += texte + '\n'

    # Reconstituer paragraphes : on colle les lignes sauf celles qui finissent par un point.
    lignes = texte_total.splitlines()
    paragraphes = []
    buffer = ""
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue
        buffer += " " + ligne
        if ligne.endswith('.') or ligne.endswith(':'):
            paragraphes.append(buffer.strip())
            buffer = ""
    if buffer:
        paragraphes.append(buffer.strip())

    texte_fusionne = "\n".join(paragraphes)
    return texte_fusionne


def splitter_texte(texte, chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(texte)


def sauvegarder_chunks(chunks, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    # Chemin dossier output
    output_dir = "supdevinci-chatbot/data/processed"

    # === Fichier TXT (site web)
    with open("supdevinci-chatbot/data/raw/site_web/supdevinci_website.txt", "r", encoding="utf-8") as f:
        raw_txt = f.read()
    clean_txt = nettoyer_texte_site_web(raw_txt)
    chunks_txt = splitter_texte(clean_txt)
    sauvegarder_chunks(chunks_txt, os.path.join(output_dir, "site_chunks.json"))
    print(f"✅ Site web : {len(chunks_txt)} chunks exportés dans {output_dir}")

    # === Fichiers PDF (règlements)
    pdfs = [
        "supdevinci-chatbot/data/raw/docs_interne/reglement_1.pdf",
        "supdevinci-chatbot/data/raw/docs_interne/reglement_2.pdf"
    ]  # Ajoute tous tes PDFs ici
    clean_pdf_text = lire_et_nettoyer_pdfs(pdfs)
    chunks_pdf = splitter_texte(clean_pdf_text)
    sauvegarder_chunks(chunks_pdf, os.path.join(output_dir, "reglement_chunks.json"))
    print(f"✅ Règlements : {len(chunks_pdf)} chunks exportés dans {output_dir}")


if __name__ == "__main__":
    main()
