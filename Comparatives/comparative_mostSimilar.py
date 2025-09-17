import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento del dataset
input_path = "comparative_csv.csv"
df = pd.read_csv(input_path)

# Liste delle parole
comparatives = df['comparative'].tolist()
positives = df['positive'].tolist()

# Inizializza matrice
matrix = np.zeros((len(comparatives), len(positives)), dtype=int)

# Loop su tutte le combinazioni: comp_i - pos_i + pos_j ≈ comp_j
for i in tqdm(range(len(comparatives)), desc="Processing rows"):
    comp_i = comparatives[i]
    pos_i = positives[i]

    try:
        for j in range(len(positives)):
            pos_j = positives[j]
            comp_j = comparatives[j]

            # Operazione vettoriale: comp_i - pos_i + pos_j
            predicted = word_vectors.most_similar(positive=[comp_i, pos_j], negative=[pos_i], topn=1)

            if predicted[0][0] == comp_j:
                matrix[i, j] = 1

    except KeyError:
        continue  # salta combinazioni con parole non presenti

# Etichette righe e colonne
index_labels = [f"{c1}-{p1}" for c1, p1 in zip(comparatives, positives)]
result_df = pd.DataFrame(matrix, index=index_labels, columns=positives)

# Visualizzazione heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(result_df, cmap=sns.color_palette(["red", "blue"]), cbar=False, linewidths=0.5, linecolor='gray')
plt.title("Comparative-Positive Comparison Matrix with most_similar")
plt.xlabel("Positive")
plt.ylabel("Comparative-Positive")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Statistiche
count_ones = np.sum(matrix == 1)
count_zeros = np.sum(matrix == 0)
total = count_ones + count_zeros
percent_ones = (count_ones / total) * 100

print(f"Conteggio '1': {count_ones}")
print(f"Conteggio '0': {count_zeros}")
print(f"Percentuale '1': {percent_ones:.2f}%")
