import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement du fichier (à adapter si le nom exact est différent)
df = pd.read_csv('rapport_evaluation_200_mon_rag_mcqu_71p.csv')

# 2. Nettoyage et calcul des fréquences
# Les scores sont sous forme '100%', '50%'. On enlève le '%' et on convertit en entier pour un tri numérique correct.
df['Score_num'] = df['Score'].str.rstrip('%').astype(int)
score_counts = df['Score_num'].value_counts().sort_index()

# 3. Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13
})

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# 4. Tracé du diagramme en barres
# On recrée les étiquettes de l'axe X en rajoutant le '%' pour l'esthétique
x_labels = [f"{score}%" for score in score_counts.index]
bars = ax.bar(x_labels, score_counts.values, 
              color='#2b5c8f', alpha=0.9, width=0.6)

# 5. Ajout des étiquettes de valeur au-dessus de chaque barre
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, # Ajustement léger de la hauteur (+0.5 au lieu de +1.5) selon vos données
            f'{int(height)}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

# 6. Habillage et titres
ax.set_title("Répartition des scores de correspondance (QCM - MCQU)", 
             pad=15, fontweight='bold', color='#1a252f')
ax.set_xlabel("Score de correspondance", labelpad=12)
ax.set_ylabel("Nombre de questions", labelpad=12)

# Ajustement de la limite supérieure de l'axe Y pour laisser de la place aux étiquettes
ax.set_ylim(0, max(score_counts.values) + max(score_counts.values)*0.15) # +15% de marge dynamique en haut

plt.tight_layout()

# 7. Sauvegarde
plt.savefig('repartition_exacte_scores_mcqu.png', dpi=300)
print("Graphique sauvegardé sous 'repartition_exacte_scores_mcqu.png'")
plt.show()