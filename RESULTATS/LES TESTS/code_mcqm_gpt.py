import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# PARAMETRES
# ==========================================================

CSV_FILE = "rapport_evaluation_200_mcqm_simple_65pourcents.csv"

OPTIONS = ["A", "B", "C", "D", "E"]

# ==========================================================
# CHARGEMENT
# ==========================================================

df = pd.read_csv(CSV_FILE)

print(f"\nNombre de questions : {len(df)}")

# ==========================================================
# CONVERSION MULTI-LABEL -> BINAIRE
# ==========================================================

y_true = []
y_pred = []

for _, row in df.iterrows():

    vraies = set(
        str(row["Vraie"]).replace(" ", "").split(",")
    )

    ia = set(
        str(row["IA"]).replace(" ", "").split(",")
    )

    for option in OPTIONS:

        y_true.append(
            1 if option in vraies else 0
        )

        y_pred.append(
            1 if option in ia else 0
        )

# ==========================================================
# MATRICE DE CONFUSION
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

# ==========================================================
# METRIQUES
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

specificity = tn / (tn + fp)

# ==========================================================
# AFFICHAGE DES RESULTATS
# ==========================================================

print("\n" + "=" * 60)
print("MATRICE DE CONFUSION")
print("=" * 60)

print(cm)

print(f"\nTP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

print("\n" + "=" * 60)
print("INDICATEURS DE PERFORMANCE")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\n" + "=" * 60)
print("RAPPORT DE CLASSIFICATION")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Option Incorrecte",
            "Option Correcte"
        ],
        zero_division=0
    )
)

# ==========================================================
# FIGURE MEMOIRE
# ==========================================================

plt.figure(figsize=(10, 8))

labels = np.array([
    [
        f"TN\n{tn}",
        f"FP\n{fp}"
    ],
    [
        f"FN\n{fn}",
        f"TP\n{tp}"
    ]
])

sns.heatmap(
    cm,
    annot=labels,
    fmt="",
    cmap="Blues",
    linewidths=2,
    linecolor="black",
    cbar=True,
    xticklabels=[
        "Prédit Négatif",
        "Prédit Positif"
    ],
    yticklabels=[
        "Réel Négatif",
        "Réel Positif"
    ]
)

plt.title(
    "Matrice de Confusion du Système RAG sans Reranking",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")

plt.tight_layout()

plt.savefig(
    "matrice_confusion_master.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# EXPORT DES METRIQUES
# ==========================================================

resultats = pd.DataFrame({
    "Mesure": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Specificity",
        "TP",
        "TN",
        "FP",
        "FN"
    ],
    "Valeur": [
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        tp,
        tn,
        fp,
        fn
    ]
})

resultats.to_csv(
    "metriques_evaluation.csv",
    index=False
)

print(
    "\nFichiers générés :"
    "\n - matrice_confusion_master.png"
    "\n - metriques_evaluation.csv"
)