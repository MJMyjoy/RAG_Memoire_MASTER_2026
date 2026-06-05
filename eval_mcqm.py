import json
import os
import re
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app_m import client, retrieve_and_rerank # On importe ton client Groq et ton retriever

# --- CONFIGURATION ---
TEST_FILE = "dataset_2/mcqm_test.json" 
LIMIT = 100 

print("⏳ Chargement du modèle NLI local pour l'évaluation...")
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def extraire_lettres(texte_ia):
    """Extrait proprement les lettres uniques (A,B,C,D,E) de la fin de la réponse."""
    match = re.search(r"RÉPONSE_FINALE\s*:\s*([A-E\s,]+)", texte_ia, re.IGNORECASE)
    if match:
        lettres = match.group(1).upper()
        return set(re.findall(r"[A-E]", lettres))
    # Fallback : si l'IA a juste écrit les lettres au milieu
    return set()

def executer_evaluation_qcm():
    if not os.path.exists(TEST_FILE):
        print(f"❌ Erreur : Fichier {TEST_FILE} introuvable.")
        return

    predictions_correctes = 0
    total_questions = 0
    rapport_details = []

    print(f"🚀 Évaluation stricte du RAG sur {LIMIT} questions de test...")

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if total_questions >= LIMIT:
                break
            line = line.strip()
            if not line: continue
                
            item = json.loads(line)
            
            # Reconstitution de la question AVEC ses options
            case = item.get("clinical_case", "")
            case_text = f"Cas clinique : {case}\n" if case else ""
            question = item["question"]
            
            # On injecte les choix pour que Llama puisse faire l'association !
            choix = f"\nA) {item.get('answer_a','')}\nB) {item.get('answer_b','')}\nC) {item.get('answer_c','')}\nD) {item.get('answer_d','')}\nE) {item.get('answer_e','')}"
            
            vraie_reponse_str = item.get("correct_answers", "").upper()
            vraies_lettres = set(re.findall(r"[A-E]", vraie_reponse_str))

            # 1. Étape du Retriever (LanceDB)
            contextes = retrieve_and_rerank(question, top_n=5)
            contexte_global = " ".join([c["text"] for c in contextes]) if contextes else "Aucun contexte."

            # 2. Prompt ÉVALUATION (Ultra-Strict pour forcer les lettres)
            prompt_eval = f"""Tu es un agent d'évaluation de QCM médicaux. 
Analyse le cas, la question et les choix fournis ci-dessous.
Tu as le droit d'utiliser tes connaissances si le contexte fourni est insuffisant.

{case_text}
Question : {question}
{choix}

Donne ton raisonnement court, puis termine EXACTEMENT par la ligne suivante :
RÉPONSE_FINALE : suivi des lettres correctes séparées par une virgule.
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Tu réponds uniquement sous le format RÉPONSE_FINALE : X, Y"},
                    {"role": "user", "content": prompt_eval}
                ],
                temperature=0.0 # Strictement déterministe
            )
            
            reponse_ia = response.choices[0].message.content
            
            # 3. Extraction et Comparaison par ensembles
            lettres_predites = extraire_lettres(reponse_ia)
            est_correct = (lettres_predites == vraies_lettres) and len(vraies_lettres) > 0
            
            if est_correct:
                predictions_correctes += 1

            # 4. Calcul NLI
            inputs = tokenizer(contexte_global, reponse_ia, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = logits.softmax(dim=1)[0].tolist()

            rapport_details.append({
                "Question": question[:40] + "...",
                "Vraie": ",".join(sorted(list(vraies_lettres))),
                "IA": ",".join(sorted(list(lettres_predites))),
                "Correct": est_correct,
                "Soutien NLI": round(probs[0], 2)
            })
            total_questions += 1

    # Rapport final
    accuracy = (predictions_correctes / total_questions) * 100
    print("\n" + "="*50)
    print("📊 NOUVEAU RAPPORT CORRIGÉ (MODE QCM)")
    print("="*50)
    print(f"🔹 Questions testées : {total_questions}")
    print(f"🎯 VRAIE Accuracy (Exact Match des options) : {accuracy:.2f} %")
    print("="*50)
    
    df = pd.DataFrame(rapport_details)
    print(df.to_string(index=False))

if __name__ == "__main__":
    executer_evaluation_qcm()
