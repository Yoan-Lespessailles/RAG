from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from time import perf_counter

import re
import pymupdf

dossier = Path(__file__).resolve().parent
fichier_pdf = dossier / "la_cloche_des_roseaux.pdf"

sections = []

with pymupdf.open(fichier_pdf) as document:
    for numero_page, page in enumerate(document, start=1):
        section_courante = None

        # Le format "dict" conserve le texte et sa mise en forme :
        # blocs, lignes, fragments de texte et tailles de police.
        contenu = page.get_text("dict", sort=True)

        for bloc in contenu["blocks"]:
            # Ne traiter que les blocs de texte.
            if bloc["type"] != 0:
                continue

            for ligne in bloc["lines"]:
                fragments = ligne["spans"]

                texte = "".join(
                    fragment["text"] for fragment in fragments
                )

                texte = " ".join(texte.split())

                if not texte:
                    continue

                # Retirer le pied de page et la pagination.
                if texte.startswith("LA CLOCHE DES ROSEAUX"):
                    continue

                if re.fullmatch(r"\d+\s*/\s*\d+", texte):
                    continue

                # Dans CE PDF, les titres de section sont en 13 points.
                # La petite marge évite une comparaison exacte de décimaux.
                est_titre_section = all(
                    12.5 <= fragment["size"] <= 13.5
                    for fragment in fragments
                    if fragment["text"].strip()
                )

                if est_titre_section:
                    # Commencer une nouvelle section.
                    section_courante = {
                        "titre": texte,
                        "lignes": [],
                        "source": fichier_pdf.name,
                        "page": numero_page,
                    }
                    sections.append(section_courante)

                elif section_courante is not None:
                    # Rattacher le texte au dernier titre rencontré.
                    section_courante["lignes"].append(texte)

                # Le texte avant la première section est ignoré :
                # numéro de chapitre, grand titre et sous-titre de page.

# Réunir les lignes du corps de chaque section.
for section in sections:
    section["texte"] = " ".join(section.pop("lignes"))

nom_modele = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(nom_modele)


def compter_tokens(texte):
    # Mesurer le texte sans les tokens spéciaux du modèle.
    return len(
        tokenizer.encode(
            texte,
            add_special_tokens=False,
            truncation=False,
        )
    )


passages = []

for section in sections:
    # Le titre fera partie du texte envoyé au modèle d’embeddings.
    prefixe = section["titre"] + "\n"
    texte_complet = prefixe + section["texte"]

    if compter_tokens(texte_complet) <= 100:
        # Garder la section entière lorsqu’elle est assez courte.
        morceaux = [section["texte"]]

    else:
        # Réserver une partie des 100 tokens pour le titre.
        # La marge de 2 tient compte des variations de tokenisation
        # lors de l’assemblage du titre et du corps.
        budget_corps = 100 - compter_tokens(prefixe) - 2

        if budget_corps <= 0:
            raise ValueError(
                f"Titre trop long : {section['titre']}"
            )

        decoupeur = RecursiveCharacterTextSplitter(
            chunk_size=budget_corps,
            chunk_overlap=min(20, budget_corps // 4),
            length_function=compter_tokens,
            separators=[". ", "; ", " ", ""],
        )

        # Découper uniquement le corps : le titre sera ajouté ensuite.
        morceaux = decoupeur.split_text(section["texte"])

    for numero, morceau in enumerate(morceaux, start=1):
        texte_passage = prefixe + morceau

        # Vérifier la longueur réelle après ajout du titre,
        # en incluant cette fois les tokens spéciaux.
        longueur_complete = len(
            tokenizer.encode(texte_passage, truncation=False)
        )

        if longueur_complete > 128:
            raise ValueError(
                f"Passage trop long : {section['titre']}, "
                f"{longueur_complete} tokens."
            )

        passages.append({
            "texte": texte_passage,
            "source": section["source"],
            "page": section["page"],
            "section": section["titre"],
            "numero": numero,
        })

print(f"\nNombre de sections : {len(sections)}")
print(f"Nombre de passages : {len(passages)}")

# Examiner tous les passages issus de la première page.
for passage in passages:
    if passage["page"] == 1:
        print(
            f"\nSection : {passage['section']}"
            f" — morceau {passage['numero']}"
        )
        print(
            f"Page : {passage['page']}"
            f" — {compter_tokens(passage['texte'])} tokens de texte"
        )
        print(passage["texte"])
        print("-" * 60)

print("\nChargement du modèle d’embeddings...", flush=True)

# Conserver le même modèle et les mêmes réglages que le test précédent.
embeddings = HuggingFaceEmbeddings(
    model_name=nom_modele,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

textes = []
metadonnees = []

for passage in passages:
    textes.append(passage["texte"])
    metadonnees.append({
        "source": passage["source"],
        "page": passage["page"],
        "section": passage["section"],
        "numero": passage["numero"],
    })

print("Création de l’index FAISS...", flush=True)

index = FAISS.from_texts(
    texts=textes,
    embedding=embeddings,
    metadatas=metadonnees,
)

print("\nChargement du modèle de reclassement...", flush=True)

# Examiner ensemble la question et chaque passage.
# Le modèle est chargé une seule fois, sur CPU.
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    device="cpu",
    max_length=512,
)

# Conserver les mêmes questions pour les deux méthodes.
questions = [
    "Quelle avance Iria verse-t-elle aux chasseurs ?",
    "Combien coûte un flacon d’huile de limaille ?",
    "Quel est le numéro de téléphone de l’auberge ?",
    "À quelles conditions les chasseurs obtiennent-ils la prime ?",
    "Quelles conditions faut-il remplir et quelle preuve faut-il présenter pour obtenir la prime ?",
    "Combien faut-il verser pour louer la barque, caution comprise ?",
    "Dans quelles conditions la caution de la barque est-elle rendue ?",
    "La lumière attire-t-elle la Veilleuse des roseaux ?",
    "Vaincre la Veilleuse au combat suffit-il à mettre fin aux attaques définitivement ?",
    "Quelles sont les trois étapes du rituel et dans quel ordre faut-il les accomplir ?",
    "Quel est le nom du frère d’Orven ?",
]

for question in questions:
    print(f"\nQuestion : {question}")

    # Sélectionner huit candidats avec notre recherche habituelle.
    debut = perf_counter()
    candidats = index.similarity_search_with_score(question, k=8)
    duree_recherche = perf_counter() - debut

    # Construire les couples question-passage à examiner.
    couples = [
        (question, document.page_content)
        for document, _ in candidats
    ]

    # Mesurer le temps supplémentaire du reclassement.
    debut = perf_counter()
    scores = reranker.predict(
        couples,
        batch_size=2,
        show_progress_bar=False,
    )
    duree_reclassement = perf_counter() - debut

    # Conserver le rang initial pour observer les déplacements.
    classement = [
        (document, float(score), rang_initial)
        for rang_initial, ((document, _), score) in enumerate(
            zip(candidats, scores),
            start=1,
        )
    ]

    # Avec ce modèle, un score plus élevé est mieux classé.
    classement.sort(key=lambda resultat: resultat[1], reverse=True)

    print("\nFAISS seul — quatre premiers résultats")

    for rang, (document, _) in enumerate(candidats[:4], start=1):
        meta = document.metadata
        print(
            f"{rang}. Page {meta['page']}"
            f" — {meta['section']} — morceau {meta['numero']}"
        )

    print("\nFAISS puis reranker — quatre premiers résultats")

    for rang, (document, score, rang_initial) in enumerate(
        classement[:4],
        start=1,
    ):
        meta = document.metadata
        print(
            f"{rang}. Page {meta['page']}"
            f" — {meta['section']} — morceau {meta['numero']}"
            f" — rang FAISS initial : {rang_initial}"
            f" — score : {score:.4f}"
        )

    print(f"\nTemps FAISS : {duree_recherche:.2f} s")
    print(f"Temps supplémentaire du reranker : {duree_reclassement:.2f} s")
    print("-" * 60)

