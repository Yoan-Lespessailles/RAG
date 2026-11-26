from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from pathlib import Path
import re

import pymupdf

dossier = Path(__file__).resolve().parent
fichier_pdf = dossier / "la_cloche_des_roseaux.pdf"

# Conserver le texte et les informations de source de chaque page.
pages = []

with pymupdf.open(fichier_pdf) as document:
    for numero, page in enumerate(document, start=1):
        lignes_utiles = []

        for ligne in page.get_text(sort=True).splitlines():
            ligne = ligne.strip()

            # Retirer le pied de page répété.
            if ligne.startswith("LA CLOCHE DES ROSEAUX  /"):
                continue

            # Retirer uniquement la pagination, par exemple « 1 / 6 ».
            if re.fullmatch(r"\d+\s*/\s*\d+", ligne):
                continue

            lignes_utiles.append(ligne)

        # Réunir les lignes de la page et remplacer les suites
        # d’espaces ou de retours à la ligne par un seul espace.
        # C’est un premier nettoyage simple pour ce PDF à une colonne.
        texte = " ".join(" ".join(lignes_utiles).split())

        if texte:
            pages.append({
                "texte": texte,
                "source": fichier_pdf.name,
                "page": numero,
            })

# Charger uniquement le tokenizer de notre modèle d’embeddings.
# Il découpe le texte en tokens, sans calculer les vecteurs.
nom_modele = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(nom_modele)


def compter_tokens(texte):
    # Ne pas tronquer : nous voulons mesurer le texte entier.
    # Les tokens spéciaux du modèle sont exclus de ce comptage.
    return len(
        tokenizer.encode(
            texte,
            add_special_tokens=False,
            truncation=False,
        )
    )


# Découper en visant au maximum 100 tokens de texte par passage.
# Les séparateurs sont essayés du plus large au plus fin.
decoupeur = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=compter_tokens,
    # Essayer de couper entre les phrases avant de couper entre les mots.
    separators=[". ", "; ", " ", ""],
)

passages = []

# Traiter les pages séparément pour conserver une source précise.
for page in pages:
    morceaux = decoupeur.split_text(page["texte"])

    for numero, morceau in enumerate(morceaux, start=1):
        # Vérifier la longueur réelle, tokens spéciaux compris.
        longueur_complete = len(
            tokenizer.encode(morceau, truncation=False)
        )

        # Notre modèle d’embeddings accepte 128 tokens au maximum.
        if longueur_complete > 128:
            raise ValueError(
                f"Passage trop long : page {page['page']}, "
                f"passage {numero}, {longueur_complete} tokens."
            )

        passages.append({
            "texte": morceau,
            "source": page["source"],
            "page": page["page"],
            "numero": numero,
        })

print(f"\nNombre de pages : {len(pages)}")
print(f"Nombre de passages : {len(passages)}")

# Afficher les cinq premiers passages en entier.
for passage in passages[:5]:
    print(
        f"\nPage {passage['page']} — passage {passage['numero']}"
        f" — {compter_tokens(passage['texte'])} tokens de texte"
    )
    print(passage["texte"])
    print("-" * 60)