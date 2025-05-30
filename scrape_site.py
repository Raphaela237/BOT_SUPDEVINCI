import requests
from bs4 import BeautifulSoup
import os
 
urls = [
    "https://www.supdevinci.fr/formations/",
    "https://www.supdevinci.fr/admission/",
    "https://www.supdevinci.fr/lecole/",
    "https://www.supdevinci.fr/campus/"
]
 
# ce script permet de recuperer un fichier txt d'informations depuis le site web de Sup de Vinci 
output_file = "supdevinci-chatbot/data/raw/site_web/supdevinci_website.txt"
os.makedirs("data", exist_ok=True)
 
with open(output_file, "w", encoding="utf-8") as f_out:
    for url in urls:
        print(f"Scraping {url} ...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
 
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
 
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_content = "\n".join(lines)
 
        f_out.write(f"\n\n===== CONTENU DE {url} =====\n\n")
        f_out.write(page_content)
 
print(f"Contenu enrichi écrit dans {output_file}") 