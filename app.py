import streamlit as st

from moteur_rag import charger_moteur, repondre


st.set_page_config(
    page_title="La Cloche des roseaux — Assistant",
    page_icon="📖",
    layout="centered",
)


@st.cache_resource
def obtenir_moteur():
    # Conserver les modèles et l’index en mémoire entre les questions.
    # Le chargement sera refait après un redémarrage du serveur.
    return charger_moteur()


st.title("La Cloche des roseaux")
st.write("Pose une question sur le scénario et consulte les passages sources.")
st.caption("Document : la_cloche_des_roseaux.pdf")


with st.form("formulaire_question"):
    question = st.text_input(
        "Ta question",
        placeholder="Exemple : combien coûte la location de la barque ?",
    )

    envoyer = st.form_submit_button("Rechercher une réponse")


if envoyer:
    question = question.strip()

    if not question:
        st.warning("Saisis une question avant de lancer la recherche.")

    else:
        # Retirer le résultat précédent avant une nouvelle tentative.
        st.session_state.pop("dernier_resultat", None)

        try:
            with st.spinner("Préparation du moteur…"):
                moteur = obtenir_moteur()

            with st.spinner("Recherche des passages et génération de la réponse…"):
                texte_reponse, sources = repondre(question, moteur)

            # Conserver le résultat pour qu’il reste affiché
            # lorsque Streamlit réexécute la page.
            st.session_state["dernier_resultat"] = {
                "question": question,
                "reponse": texte_reponse,
                "sources": sources,
            }

        except Exception as erreur:
            st.error(
                "Impossible de produire la réponse. "
                "Consulte le détail ci-dessous."
            )

            with st.expander("Détail de l’erreur"):
                st.code(str(erreur), language=None)


# Afficher le dernier résultat en dehors du formulaire.
if "dernier_resultat" in st.session_state:
    resultat = st.session_state["dernier_resultat"]

    st.subheader("Réponse")
    st.caption(f"Question : {resultat['question']}")
    st.write(resultat["reponse"])

    st.subheader("Passages transmis au modèle")
    st.caption(
        "Ces extraits permettent de vérifier la réponse ; "
        "leur sélection ne garantit pas qu’ils contiennent toute l’information."
    )

    for numero, document in enumerate(resultat["sources"], start=1):
        meta = document.metadata

        titre = (
            f"Extrait {numero} — page {meta['page']}"
            f" — {meta['section']}"
        )

        with st.expander(titre):
            st.caption(
                f"Fichier : {meta['source']} — morceau {meta['numero']}"
            )
            st.text(document.page_content)