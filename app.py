import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- CONFIGURATION GEMINI ---
# Récupère la clé dans .streamlit/secrets.toml sous le nom GEMINI_API_KEY
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Clé API Gemini manquante !")
    st.stop()

st.set_page_config(page_title="IA Catalogue AFD", page_icon="🧩")
st.title("🧩 Expert Conseil AFD")


# --- MOTEUR DE RECHERCHE (Identique à avant) ---
@st.cache_resource
def preparer_moteur_recherche():
    df = pd.read_csv("catalogue_afd.csv").fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    df['text_complet'] = "Produit: " + df['nom'] + " | Catégorie: " + df['categorie'] + " | Description: " + df[
        'description']

    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_embed.encode(df['text_complet'].tolist(), convert_to_tensor=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return df, model_embed, index


df, model_embed, index = preparer_moteur_recherche()

# --- LOGIQUE DE RÉPONSE ---
query = st.text_input("Posez votre question sur le catalogue :")

if query:
    with st.spinner("Gemini analyse le catalogue..."):
        # 1. Recherche des produits proches
        query_vector = model_embed.encode([query])
        distances, indices = index.search(np.array(query_vector).astype('float32'), k=5)

        contexte_produits = ""
        for i in indices[0]:
            p = df.iloc[i]
            contexte_produits += f"PRODUIT: {p['nom']}\nDESC: {p['description']}\nLIEN: {p['url']}\n\n"
        # Définition du caractère de l'IA
        INSTRUCTIONS_CONSEILLER = """
        Tu es un conseiller expert d'Autisme Diffusion (AFD), spécialisé dans l'accompagnement des familles. 
        Ton ton doit être :
        1. Doux et empathique : Reconnais que le quotidien des parents peut être difficile. Utilise des phrases comme "Je comprends que cela puisse être frustrant" ou "C'est une étape importante".
        2. Pédagogue : N'explique pas seulement 'quoi' acheter, mais 'pourquoi' cela aide l'enfant (ex: expliquer le bénéfice sensoriel d'un objet).
        3. Structuré : Utilise des listes à puces pour que l'information soit facile à lire pour des parents souvent fatigués ou pressés.
        4. Prudent : Rappelle que tu es une IA et que ces conseils ne remplacent pas l'avis d'un professionnel de santé.
        
        Utilise les produits du catalogue fournis pour illustrer tes conseils.
        """
        
        # Application des instructions au modèle
        # 2. Appel au modèle Gemini (Flash 1.5 ou 2.0)
        # model_gemini = (genai.GenerativeModel('gemini-flash-latest'),
        #   system_instruction=INSTRUCTIONS_CONSEILLER
        #)      
        model_gemini = genai.GenerativeModel(
            model_name='gemini-flash-latest', # <--- LA VIRGULE EST CRUCIALE ICI !
            system_instruction=INSTRUCTIONS_CONSEILLER
        )        
        prompt = f"""Tu es l'expert d'Autisme Diffusion (AFD). 
        Aide l'utilisateur en utilisant UNIQUEMENT ces produits :
        {contexte_produits}

        Question : {query}
        Réponds de manière bienveillante et donne les liens URL."""

        response = model_gemini.generate_content(prompt)

        st.markdown("---")
        st.markdown("### 💡 Réponse de l'IA :")
        st.write(response.text)
