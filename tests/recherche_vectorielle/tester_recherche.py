from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Test avec du texte en dur
passages = [
    "La médiathèque ouvre du mardi au samedi, de 10 h à 18 h.",
    "L’inscription est gratuite pour les habitants de la commune.",
    "Chaque lecteur peut emprunter jusqu’à 5 livres pendant 21 jours.",
]

# Question pour le test de recherche
question = "Combien de temps puis-je garder un livre ?"

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

# Transformer la question en vecteur, puis chercher le passage le plus proche
# k=1 signifie un seul résultat
resultats = index.similarity_search(question, k=1)

print(f"\nQuestion : {question}")

# Chaque résultat est un objet Document
# Son attribut page_content contient le texte du passage retrouvé
for document in resultats:
    print(f"\nPassage retrouvé : {document.page_content}")

