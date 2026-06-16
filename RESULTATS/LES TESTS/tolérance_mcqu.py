import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration esthétique pour un rendu "Mémoire/Publication"
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def parse_options(option_string):
    """Convertit une chaîne de caractères (ex: 'A,B,C') en ensemble (set)."""
    if pd.isna(option_string):
        return set()
    return set([opt.strip() for opt in str(option_string).split(',') if opt.strip()])

def calculate_recall(row):
    """Calcule le Recall exact : (Vraies trouvées) / (Total des Vraies)."""
    vraie_set = parse_options(row['Vraie'])
    ia_set = parse_options(row['IA'])
    
    if len(vraie_set) == 0:
        return 0.0
    
    true_positives = len(vraie_set.intersection(ia_set))
    return true_positives / len(vraie_set)

# 2. Chargement et préparation des données
# Remplacez 'mcqu.csv' par le nom exact de votre fichier
file_path = 'rapport_evaluation_200_mon_rag_mcqu_71p.csv' 
df = pd.read_csv(file_path)

# --- NOUVELLE ÉTAPE CRUCIALE ---
# Nettoyage de la colonne 'Score' : enlève le '%' et convertit en float (ex: '50%' -> 0.5)
df['Score'] = df['Score'].astype(str).str.replace('%', '', regex=False)
# On gère les cas où la valeur pourrait être vide ou non numérique après le nettoyage
df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0) / 100.0

# Remplir les valeurs nulles éventuelles dans les colonnes de réponses
df['Vraie'] = df['Vraie'].fillna('')
df['IA'] = df['IA'].fillna('')

# Calcul du Recall pur
df['Recall_Calcule'] = df.apply(calculate_recall, axis=1)

# 3. Préparation des données pour la courbe (Seuils k)
thresholds = np.linspace(0, 1, 101) # Seuils de 0.0 à 1.0
jaccard_success_rate = []
recall_success_rate = []

total_questions = len(df)

for k in thresholds:
    # Proportion de questions où le Score (Jaccard) >= k
    jaccard_rate = (df['Score'] >= k).sum() / total_questions
    jaccard_success_rate.append(jaccard_rate)
    
    # Proportion de questions où le Recall >= k
    recall_rate = (df['Recall_Calcule'] >= k).sum() / total_questions
    recall_success_rate.append(recall_rate)

# 4. Création de la figure
plt.figure(figsize=(10, 6))

plt.plot(thresholds, recall_success_rate, label='Rappel (Sensibilité)', color='#1f77b4', linewidth=2.5)
plt.plot(thresholds, jaccard_success_rate, label='Score de correspondance (Jaccard)', color='#ff7f0e', linewidth=2.5, linestyle='--')

# Remplissage sous la courbe pour l'esthétique
plt.fill_between(thresholds, recall_success_rate, alpha=0.1, color='#1f77b4')
plt.fill_between(thresholds, jaccard_success_rate, alpha=0.1, color='#ff7f0e')

# 5. Habillage du graphique
plt.title("Performance globale de l'IA selon le seuil d'exigence $k$", fontsize=14, weight='bold', pad=15)
plt.xlabel("Seuil de performance minimum exigé ($k$)", fontsize=12)
plt.ylabel("Proportion du jeu de données (Taux de succès)", fontsize=12)

# Ajustement des axes
plt.xlim(0, 1)
plt.ylim(0, 1.05)

# Formatage des ticks de l'axe Y en pourcentages
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.legend(loc='lower left', frameon=True, shadow=True)
plt.tight_layout()

# Sauvegarde de l'image en haute résolution pour le mémoire
plt.savefig('courbe_performance_k_mcqu.png', dpi=300)

# Affichage direct
plt.show()

# 6. Affichage de quelques statistiques descriptives dans la console
print("--- Statistiques Descriptives ---")
print(f"Moyenne du Score (Jaccard) : {df['Score'].mean():.2f}")
print(f"Moyenne du Rappel calculé  : {df['Recall_Calcule'].mean():.2f}")
print(f"Questions parfaites (Jaccard = 1.0) : {(df['Score'] == 1.0).sum()} / {total_questions}")
print(f"Questions avec toutes les vraies réponses trouvées (Recall = 1.0) : {(df['Recall_Calcule'] == 1.0).sum()} / {total_questions}")