from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Test avec du texte en dur
passages = [
    "La médiathèque ouvre du mardi au samedi, de 10 h à 18 h.",
    "L’inscription est gratuite pour les habitants de la commune.",
    "Chaque lecteur peut emprunter jusqu’à 5 livres pendant 21 jours.",
    "L'inscription est de 6€ pour les adultes n'habitant pas dans la commune"
]

print("Chargement du modèle d’embeddings...")

# Charger le modèle qui transforme les textes en vecteurs
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Création de l’index FAISS...")

# Transformer les passages en vecteurs et les placer dans un index en mémoire
# FAISS permet de rechercher les vecteurs les plus proches
index = FAISS.from_texts(
    texts=passages,
    embedding=embeddings,
)

# Tester différents sujets, puis une information absente du corpus.
questions = [
    "Combien de temps puis-je garder un livre ?",
    "Est-ce que l’inscription est payante ?",
    "À quelle heure la médiathèque ouvre-t-elle ?",
    "Quel est le numéro de téléphone de la médiathèque ?",
]

for question in questions:
    print(f"\nQuestion : {question}")

    # Retrouver les deux passages les plus proches et leurs distances.
    resultats = index.similarity_search_with_score(question, k=2)

    # Chaque résultat contient désormais deux éléments :
    # le Document et sa distance par rapport à la question.
    for rang, (document, distance) in enumerate(resultats, start=1):
        print(f"\nRésultat {rang} — distance : {distance:.4f}")
        print(document.page_content)

    print("\n" + "-" * 60)