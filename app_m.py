import streamlit as st
import json
import os
import lancedb
import pyarrow as pa
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import edsnlp

# ==========================================
# 1. CONFIGURATION ET INITIALISATION
# ==========================================
st.set_page_config(page_title="MediQAl RAG Assistant", layout="wide")

# Clé API Groq (À définir dans tes variables d'environnement ou ici directement)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")    # Remplacez os.environ.get("GROQ_API_KEY") par la clé
client = Groq(api_key=GROQ_API_KEY)

# Dossier des données
DATA_FOLDER = "dataset_1"
DB_PATH = "./lancedb_medical"

@st.cache_resource
def load_models():
    """Charge les modèles locaux une seule fois en cache."""
    # 1. Bi-Encoder pour LanceDB (Recherche vectorielle)
    embedder = SentenceTransformer("BAAI/bge-m3")
    
    # 2. Cross-Encoder pour le Reranking
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    
    # 3. NLI pour la détection d'hallucination / certitude
    nli_tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    
    # 4. EDS-NLP pour la granularité (découpage en phrases)
    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    
    return embedder, reranker, nli_tokenizer, nli_model, nlp

embedder, reranker, nli_tokenizer, nli_model, nlp = load_models()

# ==========================================
# 2. PRÉPARATION DES DONNÉES ET CHUNKING
# ==========================================

def parse_mediqai_json(file_path):
    """Lit et transforme les JSON Lines en textes clairs pour l'embedding."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: # Ignore les lignes vides pour éviter les plantages
                continue
                
            item = json.loads(line) # Parse chaque ligne individuellement
            
            case = item.get("clinical_case", "")
            # ... (Le reste de ton code en dessous reste exactement le même)
            case_text = f"Cas clinique : {case}\n" if case else ""
            question = item.get("question", "")

            subject = item.get("medical_subject", "Médecine générale")
        
            # Reconstruire les réponses correctes en texte naturel (pas juste les lettres)
            correct_text = ""
            if "correct_answers" in item and "answer_a" in item:
                correct_letters = [letter.strip() for letter in item["correct_answers"].split(',')]
                correct_phrases = []
                for letter in correct_letters:
                    key = f"answer_{letter.lower()}"
                    if key in item:
                        correct_phrases.append(item[key])
                correct_text = "La ou les bonnes réponses sont : " + " ; ".join(correct_phrases)
            elif "answer" in item: # Pour oeq_test
                correct_text = f"Réponse attendue : {item['answer']}"
            
            # Chunk final enrichi
            chunk = f"[{subject}] {case_text}Question : {question}\n{correct_text}"
            documents.append({"id": str(item["id"]), "text": chunk, "subject": subject})
        
    return documents

def setup_database():
    """Initialise LanceDB uniquement si nécessaire."""
    db = lancedb.connect(DB_PATH)
    table_name = "mediqai_knowledge"
    
    if table_name in db.table_names():
        return db.open_table(table_name)
        
    st.info("Création de la base vectorielle en cours (cela peut prendre quelques minutes la première fois)...")
    
    all_docs = []
    for filename in ["mcqm_train.json", "mcqu_train.json", "oeq_test.json"]:
        path = os.path.join(DATA_FOLDER, filename)
        if os.path.exists(path):
            all_docs.extend(parse_mediqai_json(path))
            
    if not all_docs:
        st.error("Aucun fichier de données trouvé dans 'dataset_1'.")
        return None
        
    # Création des embeddings
    texts = [doc["text"] for doc in all_docs]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # Structure pour LanceDB
    data_to_insert = pa.Table.from_arrays(
        [
            pa.array([doc["id"] for doc in all_docs]),
            pa.array([doc["text"] for doc in all_docs]),
            pa.array([doc["subject"] for doc in all_docs]),
            pa.array(embeddings.tolist()) # LanceDB a besoin de listes
        ],
        names=["id", "text", "subject", "vector"]
    )
    
    table = db.create_table(table_name, data=data_to_insert)
    st.success("Base vectorielle LanceDB créée avec succès !")
    return table

table = setup_database()

# ==========================================
# 3. PIPELINE RAG (Retriever, Reranker, LLM, NLI)
# ==========================================

def retrieve_and_rerank(query, top_k=15, top_n=5):
    """Recherche hybride vectorielle puis reranking précis."""
    query_embedding = embedder.encode(query).tolist()
    
    # 1. Retrieval BGE-M3 (LanceDB)
    results = table.search(query_embedding).limit(top_k).to_list()
    
    if not results:
        return []
        
    # 2. Cross-Encoding (BGE-Reranker)
    pairs = [[query, res["text"]] for res in results]
    scores = reranker.predict(pairs)
    
    for i, res in enumerate(results):
        res["rerank_score"] = float(scores[i])
        
    # Trier par score de reranking décroissant
    reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    return reranked_results

def check_hallucination_nli(premise, hypothesis):
    """Évalue si la phrase (hypothesis) est déduite du contexte (premise)."""
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = logits.softmax(dim=1)[0].tolist()
    # Classes typiques MoritzLaurer : 0: Entailment, 1: Neutral, 2: Contradiction
    return {"entailment": probs[0], "neutral": probs[1], "contradiction": probs[2]}

def generate_and_verify(query, contexts):
    """Génère la réponse avec Llama 4 et l'évalue phrase par phrase."""
    
    context_text = "\n\n".join([f"Source {i+1}: {ctx['text']}" for i, ctx in enumerate(contexts)])
    

    system_prompt = """Tu es un assistant médical expert francophone. Ta règle absolue est la rigueur et la prudence.
Règles strictes :
1. Salutations : Si l'utilisateur dit bonjour, réponds poliment.
2. Basé EXCLUSIVEMENT sur les sources : Utilise UNIQUEMENT les sources fournies pour répondre. Ne cite pas de lettres (A, B, C), mais explique les concepts.
3. Absence d'information (VITAL) : Si la réponse à la question ne se trouve pas CLAIREMENT et EXPLICITEMENT dans le contexte médical fourni, tu DOIS répondre EXACTEMENT : "Je suis désolé, mais les documents locaux à ma disposition ne contiennent pas cette information. Pour des raisons de sécurité médicale, je ne peux pas formuler de supposition."
4. INTERDICTION ABSOLUE d'utiliser tes propres connaissances pré-entraînées, de deviner ou de déduire au-delà de ce qui est écrit noir sur blanc dans les sources.
5. Formate TOUJOURS ta réponse sous forme de phrases déclaratives complètes (Sujet + Verbe + Complément). INTERDICTION ABSOLUE d'utiliser des listes à puces (•, -, *, etc.) ou des mots isolés. Chaque complication ou fait médical doit faire l'objet d'une phrase complète (ex: "La diverticulose sigmoïdienne peut se compliquer d'une hémorragie digestive.") afin de permettre sa validation logique.
"""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte médical fourni :\n{context_text}\n\nQuestion du patient/médecin : {query}"}
    ]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.1, # Très bas pour éviter les hallucinations
        max_tokens=1024
    )
    
    llm_answer = response.choices[0].message.content
    
    # --- Granularité et Vérification NLI avec EDS-NLP ---
    doc = nlp(llm_answer)
    sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 10]
    
    verified_output = []
    combined_context = " ".join([ctx["text"] for ctx in contexts])
    
    for sent in sentences:
        # On ignore la vérification pour les phrases de salutation ou de limitation
        if "bonjour" in sent.lower() or "les documents fournis" in sent.lower() or "mes connaissances" in sent.lower():
            verified_output.append({"text": sent, "status": "bypassed"})
            continue
            
        nli_scores = check_hallucination_nli(combined_context, sent)
        
        if nli_scores["entailment"] > 0.60:
            status = "Soutenu (Source trouvée)"
            color = "green"
        elif nli_scores["contradiction"] > 0.40:
            status = "Contradiction (Alerte Hallucination)"
            color = "red"
        else:
            status = "Neutre / Connaissance Externe"
            color = "orange"
            
        verified_output.append({"text": sent, "status": status, "scores": nli_scores, "color": color})

    return llm_answer, verified_output, contexts

# ==========================================
# 4. INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.title("🩺 MediQAl - RAG Anti-Hallucination")

query = st.text_input("Posez votre question médicale (ou testez une salutation / un cas complexe) :")

if st.button("Analyser") and query:
    with st.spinner("Recherche et réflexion en cours..."):
        # 1. Retrieval
        contexts = retrieve_and_rerank(query)
        
        # 2. Generation & NLI Check
        if not contexts and "bonjour" not in query.lower() and "salut" not in query.lower():
            st.warning("Aucune source locale pertinente trouvée. L'IA utilisera ses connaissances générales.")
            contexts = [{"text": "Aucune information locale."}]
            
        raw_answer, verified_data, used_contexts = generate_and_verify(query, contexts)
        
        # 3. Affichage
        st.subheader("🤖 Réponse de l'IA")
        st.write(raw_answer)
        
        st.subheader("📊 Transparence et Analyse d'Incertitude (NLI)")
        for item in verified_data:
            if item["status"] == "bypassed":
                continue
            
            st.markdown(f"<span style='color:{item['color']}'><b>{item['status']}</b></span> : {item['text']}", unsafe_allow_html=True)
            scores = item["scores"]
            st.caption(f"Scores NLI -> Soutien (Entailment): {scores['entailment']:.2f} | Neutre: {scores['neutral']:.2f} | Contradiction: {scores['contradiction']:.2f}")

        with st.expander("📄 Voir les documents sources utilisés (Citations strictes)"):
            for i, ctx in enumerate(used_contexts):
                st.info(f"**Source {i+1}** (Score de pertinence: {ctx.get('rerank_score', 0):.2f}) :\n{ctx['text']}")
