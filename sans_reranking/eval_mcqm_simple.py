import json
import os
import re
import torch
import time
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from groq import RateLimitError  # Importation pour gérer les limites de requêtes

# Importations depuis votre fichier local app_simple.py
from app_simple import client, retrieve, device

# --- CONFIGURATION ---

TEST_FILE = "dataset_2/mcqm_test.json" 
TRAIN_FILE = "dataset_1/mcqm_train.json"  # 1. Ajout du fichier train
LIMIT = 200                               # 2. Passage à 200 questions
OUTPUT_CSV = "rapport_evaluation_200.csv" # 3. Sortie mise à jour


print(f"⏳ Activation du Device : {device.upper()}")
print("⏳ Chargement du modèle NLI local pour l'évaluation sur GPU...")

tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli").to(device)

def extraire_lettres(texte_ia):
    """Extrait proprement les lettres uniques (A,B,C,D,E) de la fin de la réponse."""
    match = re.search(r"RÉPONSE_FINALE\s*:\s*([A-E\s,]+)", texte_ia, re.IGNORECASE)
    if match:
        lettres = match.group(1).upper()
        return set(re.findall(r"[A-E]", lettres))
    return set()

def executer_evaluation_qcm():

    # Vérification des deux fichiers
    for chemin in [TEST_FILE, TRAIN_FILE]:
        if not os.path.exists(chemin):
            print(f"❌ Erreur : Fichier {chemin} introuvable.")
            return

    score_total = 0.0
    total_questions = 0
    rapport_details = []

    print(f"🚀 Évaluation stricte du RAG sur {LIMIT} questions de test (GPU accéléré)...")

    # Pré-chargement combiné (100 test + 100 train)
    toutes_les_lignes = []
    for chemin in [TEST_FILE, TRAIN_FILE]:
        with open(chemin, 'r', encoding='utf-8') as f:
            for idx, l in enumerate(f):
                if idx >= 100: break  # On s'arrête à 100 par fichier
                if l.strip(): toutes_les_lignes.append(l)

    for _ in [1]:  # Astuce pour maintenir l'alignement de tes 8 espaces
        for line in toutes_les_lignes:  # Maintient l'alignement de tes 12 espaces

            if total_questions >= LIMIT:
                break
            line = line.strip()
            if not line: 
                continue
                
            item = json.loads(line)
            
            # Reconstitution de la question AVEC ses options
            case = item.get("clinical_case", "")
            case_text = f"Cas clinique : {case}\n" if case else ""
            question = item["question"]
            
            choix = f"\nA) {item.get('answer_a','')}\nB) {item.get('answer_b','')}\nC) {item.get('answer_c','')}\nD) {item.get('answer_d','')}\nE) {item.get('answer_e','')}"
            
            vraie_reponse_str = item.get("correct_answers", "").upper()
            vraies_lettres = set(re.findall(r"[A-E]", vraie_reponse_str))

            # 1. Étape du Retriever (LanceDB via app_simple.py)
            contextes = retrieve(question, top_k=5)
            contexte_global = " ".join([c["text"] for c in contextes]) if contextes else "Aucun contexte."

            # 2. Prompt ÉVALUATION Strict
            prompt_eval = f"""Tu es un agent d'évaluation de QCM médicaux. 
Analyse le cas, la question et les choix fournis ci-dessous.
Tu l'autorisation d'utiliser tes connaissances si le contexte fourni est insuffisant.

{case_text}
Question : {question}
{choix}

Donne ton raisonnement court, puis termine EXACTEMENT par la ligne suivante :
RÉPONSE_FINALE : suivi des lettres correctes séparées par une virgule.
"""

            # Boucle de résilience face aux limites de taux Groq (Rate Limit)
            reponse_ia = ""
            for tentative in range(5):
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {"role": "system", "content": "Tu réponds uniquement sous le format RÉPONSE_FINALE : X, Y"},
                            {"role": "user", "content": prompt_eval}
                        ],
                        temperature=0.0
                    )
                    reponse_ia = response.choices[0].message.content
                    break  # Succès, on sort de la boucle de tentative
                except RateLimitError:
                    temps_attente = (tentative + 1) * 4
                    print(f"\n⚠️ Rate Limit Groq atteint. Pause forcée de {temps_attente} secondes (Question {total_questions + 1})...")
                    time.sleep(temps_attente)
                except Exception as e:
                    print(f"\n❌ Erreur API : {str(e)}. Nouvelle tentative...")
                    time.sleep(2)
            
            if not reponse_ia:
                print(f"❌ Impossible d'obtenir une réponse pour la question {total_questions + 1}. Passage à la suivante.")
                continue
            
            # 3. Extraction et Comparaison (Score Partiel / Jaccard)
            lettres_predites = extraire_lettres(reponse_ia)
            
            if len(vraies_lettres) > 0 or len(lettres_predites) > 0:
                intersection = vraies_lettres.intersection(lettres_predites)
                union = vraies_lettres.union(lettres_predites)
                score_qcm = len(intersection) / len(union) if len(union) > 0 else 0.0
            else:
                score_qcm = 0.0
                
            score_total += score_qcm

            # 4. Calcul NLI sur GPU
            inputs = tokenizer(contexte_global, reponse_ia, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = logits.softmax(dim=1)[0].tolist()

            rapport_details.append({
                "ID": item.get("id", total_questions),
                "Question": question[:50] + "...",
                "Vraie": ",".join(sorted(list(vraies_lettres))),
                "IA": ",".join(sorted(list(lettres_predites))),
                "Score": round(score_qcm, 2), # Affiche un score comme 0.67 ou 1.0
                "Soutien NLI": round(probs[0], 2)
            })
            
            total_questions += 1
            
            # Indicateur visuel de progression toutes les 50 questions
            if total_questions % 50 == 0:
                print(f"Progression : {total_questions}/{LIMIT} questions traitées...")

    # Génération des rapports d'analyses
    accuracy = (score_total / total_questions) * 100 if total_questions > 0 else 0
    print("\n" + "="*50)
    print(f"📊 RAPPORT FINAL DE PERFORMANCES ({LIMIT} QCM)")
    print("="*50)
    print(f"🔹 Questions validées : {total_questions}")
    print(f"🎯 ACCURACY MOYENNE (Score Partiel) : {accuracy:.2f} %")
    print("="*50)
    
    # Transformation en DataFrame & Sauvegarde en CSV
    df = pd.DataFrame(rapport_details)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"💾 Rapport complet sauvegardé dans le fichier : {OUTPUT_CSV}")
    
    print("\n👀 Aperçu des 15 premières lignes :")
    print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    executer_evaluation_qcm()
