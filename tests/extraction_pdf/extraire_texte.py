from pathlib import Path  # Pour manipuler les chemins de fichiers
import pymupdf  # Pour ouvrir le PDF et en extraire le texte

# Récupère le dossier qui contient ce script Python
# Le PDF sera cherché dans ce dossier, même si on lance le script ailleurs
dossier = Path(__file__).resolve().parent

# PDF du scénario, placé dans le même dossier que ce script.
fichier_pdf = dossier / "la_cloche_des_roseaux.pdf"

# Ouvre le PDF. Le bloc with ferme automatiquement le document à la fin
with pymupdf.open(fichier_pdf) as document:
    # Parcourt les pages et les numérote à partir de 1
    for numero, page in enumerate(document, start=1):
        # Affiche le numéro de page, précédé d’un saut de ligne (\n)
        print(f"\n--- Page {numero} ---")

        # Extrait le texte et retire les espaces et sauts de ligne aux extrémités
        texte = page.get_text().strip()

        # Affiche le texte extrait ou un message si la chaîne est vide
        print(texte or "[Aucun texte extrait]")