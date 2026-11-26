from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

print("\nChargement du modèle d’embeddings...", flush=True)

# Utiliser le même modèle que celui choisi pour le tokenizer.
embeddings = HuggingFaceEmbeddings(
    model_name=nom_modele,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

textes = []
metadonnees = []

# Préparer deux listes qui se correspondent :
# textes[0] possède les informations de metadonnees[0], etc.
for passage in passages:
    textes.append(passage["texte"])

    metadonnees.append({
        "source": passage["source"],
        "page": passage["page"],
        "numero": passage["numero"],
    })

print("Calcul des vecteurs et création de l’index FAISS...", flush=True)

# Calculer un vecteur par passage et conserver les sources associées.
index = FAISS.from_texts(
    texts=textes,
    embedding=embeddings,
    metadatas=metadonnees,
)

# Examiner où le passage du contrat se situe dans le classement.
question = "Quelle avance Iria verse-t-elle aux chasseurs ?"

print("\nDiagnostic : classement des cinq premiers passages")

for rang, (document, distance) in enumerate(
    index.similarity_search_with_score(question, k=5),
    start=1,
):
    print(
        f"\n{rang}. Page {document.metadata['page']}"
        f" — passage {document.metadata['numero']}"
        f" — distance : {distance:.4f}"
    )
    print(document.page_content)

# Examiner le classement complet pour cette seule question.
resultats = index.similarity_search_with_score(
    "Quelle avance Iria verse-t-elle aux chasseurs ?",
    k=len(passages),
)

# Afficher uniquement le passage contenant le paiement du contrat.
for rang, (document, distance) in enumerate(resultats, start=1):
    if (
        document.metadata["page"] == 1
        and document.metadata["numero"] == 3
    ):
        print(
            f"\nPassage du paiement : rang {rang}/{len(resultats)}"
            f" — distance : {distance:.4f}"
        )
        print(document.page_content)