import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement et calcul des fréquences
df = pd.read_csv('rapport_evaluation_200_mon_rag_mcqm_69p.csv')
score_counts = df['Score'].value_counts().sort_index()

# 2. Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13
})

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# 3. Tracé du diagramme en barres
bars = ax.bar(score_counts.index.astype(str), score_counts.values, 
              color='#2b5c8f', alpha=0.9, width=0.6)

# 4. Ajout des étiquettes de valeur au-dessus de chaque barre
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + 1.5, 
            f'{int(height)}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# 5. Habillage et titres
ax.set_title("Répartition des scores de correspondance par valeurs exactes de QCM", 
             pad=15, fontweight='bold', color='#1a252f')
ax.set_xlabel("Score de correspondance (Valeurs discrètes)", labelpad=12)
ax.set_ylabel("Nombre de questions", labelpad=12)

# Ajustement de la limite supérieure de l'axe Y pour laisser de la place aux étiquettes
ax.set_ylim(0, max(score_counts.values) + 8)

plt.tight_layout()

# 6. Sauvegarde
plt.savefig('repartition_exacte_scores.png', dpi=300)
print("Graphique de l'Option B sauvegardé sous 'repartition_exacte_scores.png'")
plt.show()
