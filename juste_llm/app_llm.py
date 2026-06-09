import streamlit as st
import os
from groq import Groq

# ==========================================
# 1. CONFIGURATION ET INITIALISATION
# ==========================================
# Clé API Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 2. PIPELINE LLM (Génération directe)
# ==========================================

def generate_response(query):
    """Génère la réponse avec Llama 4 en s'appuyant uniquement sur ses poids (sans RAG)."""
    
    system_prompt = """Tu es un assistant médical expert francophone. Ta règle absolue est la rigueur et la prudence.
Règles strictes :
1. Salutations : Si l'utilisateur dit bonjour, réponds poliment.
2. Expertise : Utilise tes connaissances médicales pré-entraînées pour répondre avec précision à la question. Ne tronque jamais les termes médicaux.
3. Avertissement : Rappelle toujours subtilement que tu es une IA et que ta réponse ne remplace pas une consultation médicale.
4. Formate TOUJOURS ta réponse sous forme de phrases déclaratives complètes (Sujet + Verbe + Complément). INTERDICTION ABSOLUE d'utiliser des listes à puces (•, -, *, etc.) ou des mots isolés. Chaque complication ou fait médical doit faire l'objet d'une phrase complète.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question du patient/médecin : {query}"}
    ]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.1,
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# ==========================================
# 3. INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
if __name__ == "__main__":
    st.set_page_config(page_title="MediQAl Assistant", layout="wide")
    st.title("🩺 MediQAl - Assistant LLM Simple")

    query = st.text_input("Posez votre question médicale :")

    if st.button("Analyser") and query:
        with st.spinner("Génération de la réponse..."):
            # Appel direct à l'API sans recherche préalable
            raw_answer = generate_response(query)
            
            st.subheader("🤖 Réponse de l'IA")
            st.write(raw_answer)
