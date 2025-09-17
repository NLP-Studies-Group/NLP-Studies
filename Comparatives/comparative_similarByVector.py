import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento del dataset
input_path = "comparative_csv.csv"
df = pd.read_csv(input_path)

comparatives = df['comparative'].tolist()
positives = df['positive'].tolist()

matrix = np.zeros((len(comparatives), len(positives)), dtype=int)

for i in tqdm(range(len(comparatives)), desc="Processing rows"):
    comp_i = comparatives[i]
    pos_i = positives[i]

    try:
        comp_i_vec = word_vectors[comp_i]
        pos_i_vec = word_vectors[pos_i]

        for j in range(len(positives)):
            pos_j = positives[j]
            comp_j = comparatives[j]

            pos_j_vec = word_vectors[pos_j]

            # Vettore risultato analogia: comp_i - pos_i + pos_j
            target_vec = comp_i_vec - pos_i_vec + pos_j_vec

            # Trova parola più simile al vettore risultato
            predicted = word_vectors.similar_by_vector(target_vec, topn=1)[0][0]

            if predicted == comp_j:
                matrix[i, j] = 1

    except KeyError:
        continue  # salta parole non presenti nel vocabolario

# Etichette righe e colonne
index_labels = [f"{c}-{p}" for c, p in zip(comparatives, positives)]
result_df = pd.DataFrame(matrix, index=index_labels, columns=positives)

# Visualizzazione heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(result_df, cmap=sns.color_palette(["red", "blue"]), cbar=False, linewidths=0.5, linecolor='gray')
plt.title("Comparative-Positive Comparison Matrix with similar_by_vector")
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
