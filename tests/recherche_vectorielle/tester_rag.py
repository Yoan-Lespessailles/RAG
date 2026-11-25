from langchain_ollama import ChatOllama
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

# Préparer la connexion au modèle déjà installé dans Ollama.
# Le modèle sera sollicité lors de l’appel à invoke().
llm = ChatOllama(
    model="qwen3:4b-instruct",
    base_url="http://127.0.0.1:11434",
    temperature=0,
    num_ctx=2048,
    num_predict=200,
)

# Commencer par deux cas :
# une réponse nécessitant deux passages et une information absente.
questions = [
    "Est-ce que l’inscription est payante ?",
    "Quel est le numéro de téléphone de la médiathèque ?",
]

for question in questions:
    print(f"\nQuestion : {question}")

    # Rechercher les deux passages les plus proches.
    resultats = index.similarity_search_with_score(question, k=2)

    # Construire le contexte à transmettre au LLM.
    # Les distances restent affichées pour nous, mais ne sont pas envoyées.
    extraits = []

    for rang, (document, distance) in enumerate(resultats, start=1):
        print(f"\nPassage {rang} — distance : {distance:.4f}")
        print(document.page_content)

        extraits.append(f"Passage {rang} : {document.page_content}")

    # Réunir les passages dans une seule chaîne de caractères.
    contexte = "\n\n".join(extraits)

    # Le message système définit les règles de réponse.
    consigne = (
        "Réponds en français, brièvement, uniquement à partir des passages "
        "fournis. Tiens compte des conditions et des différents publics. "
        "N’invente aucune information. "
        "Si les passages ne permettent pas de répondre, écris exactement : "
        "Information absente des passages."
    )

    # Le message utilisateur contient les extraits et la question.
    message = (
        f"Passages documentaires :\n{contexte}\n\n"
        f"Question : {question}"
    )

    print("\nGénération de la réponse avec Qwen...", flush=True)

    # Envoyer une nouvelle requête indépendante pour chaque question.
    reponse = llm.invoke([
        ("system", consigne),
        ("human", message),
    ])

    # Afficher le texte produit par le modèle.
    print(f"\nRéponse de Qwen : {reponse.content}")
    print("\n" + "-" * 60)