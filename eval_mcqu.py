import json
import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Importation sécurisée depuis le fichier app_m de votre espace Colab
from app_m import client, retrieve_and_rerank 

# --- CONFIGURATION VALEURS ÉLEVÉES ---

TEST_FILE_1 = "dataset_2/mcqu_test.json" 
TEST_FILE_2 = "dataset_1/mcqu_train.json"
LIMIT_PER_FILE = 100 
OUTPUT_CSV = "rapport_evaluation_200.csv"


# Configuration GPU active
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Chargement du modèle NLI local sur l'accélérateur matériel : {device.upper()}...")

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
    predictions_correctes = 0.0
    total_questions = 0
    rapport_details = []

    # Lecture préalable pour compter et limiter proprement
    questions_a_traiter = []
    fichiers_a_tester = [TEST_FILE_1, TEST_FILE_2]

    for fichier in fichiers_a_tester:
        if not os.path.exists(fichier):
            print(f"❌ Erreur : Fichier {fichier} introuvable. On passe au suivant.")
            continue
            
        compteur = 0
        with open(fichier, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions_a_traiter.append(json.loads(line))
                    compteur += 1
                # On s'arrête à 100 pour CE fichier spécifique
                if compteur >= LIMIT_PER_FILE:
                    break


    print(f"🚀 Évaluation stricte du RAG lancée sur {len(questions_a_traiter)} questions QCM...")

    # Utilisation de tqdm pour éviter le gel d'affichage dans Colab
    for item in tqdm(questions_a_traiter, desc="Traitement des QCM"):
        case = item.get("clinical_case", "")
        case_text = f"Cas clinique : {case}\n" if case else ""
        question = item["question"]
        
        choix = f"\nA) {item.get('answer_a','')}\nB) {item.get('answer_b','')}\nC) {item.get('answer_c','')}\nD) {item.get('answer_d','')}\nE) {item.get('answer_e','')}"
        
        vraie_reponse_str = item.get("correct_answers", "").upper()
        vraies_lettres = set(re.findall(r"[A-E]", vraie_reponse_str))

        # 1. Étape du Retriever (LanceDB GPU-backed)
        contextes = retrieve_and_rerank(question, top_n=5)
        contexte_global = " ".join([c["text"] for c in contextes]) if contextes else "Aucun contexte."

        # 2. Prompt ÉVALUATION (Appel Groq optimisé)
        prompt_eval = f"""Tu es un agent d'évaluation de QCM médicaux. 
Analyse le cas, la question et les choix fournis ci-dessous.
Tu as le droit d'utiliser tes connaissances si le contexte fourni est insuffisant.

{case_text}
Question : {question}
{choix}

Donne ton raisonnement court, puis termine EXACTEMENT par la ligne suivante :
RÉPONSE_FINALE : suivi des lettres correctes séparées par une virgule.
"""

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Tu es un expert médical. Analyse la question et le contexte fourni. "
                            "Sélectionne la ou les deux meilleures lettres maximum qui te semblent correctes. "
                            "Tu devez répondre STRICTEMENT sous le format : RÉPONSE_FINALE : X, Y (ou juste X s'il n'y en a qu'une)."
                        )
                    },
                    {"role": "user", "content": prompt_eval}
                ],
                temperature=0.0
            )
            reponse_ia = response.choices[0].message.content
        except Exception as e:
            print(f"\n⚠️ Erreur lors de l'appel API Groq : {e}")
            reponse_ia = "RÉPONSE_FINALE : Erreur"

        
        # 3. Extraction et Comparaison proportionnelle (Indice de Jaccard)
        lettres_predites = extraire_lettres(reponse_ia)
        
        intersection = vraies_lettres.intersection(lettres_predites)
        union = vraies_lettres.union(lettres_predites)
        
        # Calcul du score proportionnel
        if len(union) > 0:
            score_question = len(intersection) / len(union)
        elif len(vraies_lettres) == 0 and len(lettres_predites) == 0:
            score_question = 1.0 # Cas rare : aucune réponse attendue ni donnée
        else:
            score_question = 0.0
            
        predictions_correctes += score_question # On ajoute le pourcentage (ex: 0.5) au lieu de 1


        # 4. Calcul NLI sur le GPU T4
        inputs = tokenizer(contexte_global, reponse_ia, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Tenseurs transférés sur la VRAM
        
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = logits.softmax(dim=1)[0].tolist()

        rapport_details.append({
            "Question": question[:50] + "...",
            "Vraie": ",".join(sorted(list(vraies_lettres))),
            "IA": ",".join(sorted(list(lettres_predites))),
            "Score": f"{score_question * 100:.0f}%", # Remplace la colonne 'Correct'
            "Soutien NLI": round(probs[0], 2)
        })
        total_questions += 1


    # Synthèse des résultats
    accuracy = (predictions_correctes / total_questions) * 100
    print("\n" + "="*50)
    print("📊 RAPPORT FINAL DE VALIDATION (100 QUESTIONS)")
    print("="*50)
    print(f"🔹 Questions effectivement testées : {total_questions}")
    print(f"🎯 Accuracy (Score proportionnel moyen) : {accuracy:.2f} %")
    print("="*50)

    
    # Génération et sauvegarde du DataFrame complet
    df = pd.DataFrame(rapport_details)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"💾 Rapport complet sauvegardé sous '{OUTPUT_CSV}' dans votre explorateur Colab.")
    
    # Affichage d'un aperçu des 10 premières lignes
    print("\n👀 Aperçu des 10 premiers résultats :")
    print(df.head(10).to_string(index=False))

# Lancement de l'évaluation globale
executer_evaluation_qcm()
