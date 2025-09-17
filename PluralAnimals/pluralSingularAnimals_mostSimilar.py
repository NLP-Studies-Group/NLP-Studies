import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_path = "plural_singular_animals.csv"
df = pd.read_csv(input_path)
plural = df['plural'].tolist()
singular = df['singular'].tolist()

# inizializzo matrice
matrix = np.zeros((len(plural), len(singular)), dtype=int)

# loop su tutte le combinazioni: comp_i - pos_i + pos_j ≈ comp_j
for i in tqdm(range(len(plural)), desc="Processing rows"):
    comp_i = plural[i]
    pos_i = singular[i]

    try:
        for j in range(len(singular)):
            pos_j = singular[j]
            comp_j = plural[j]

            # operazione vettoriale: comp_i - pos_i + pos_j
            predicted = word_vectors.most_similar(positive=[comp_i, pos_j], negative=[pos_i], topn=1)

            if predicted[0][0] == comp_j:
                matrix[i, j] = 1

    except KeyError:
        print(f"Parola non trovata nel vocabolario")

col_sums = matrix.sum(axis=0)
sorted_indices = np.argsort(-col_sums)

# riordino la matrice mantenendo la diagonale
matrix_sorted = matrix[np.ix_(sorted_indices, sorted_indices)]

# riordino le liste di parole corrispondenti
sorted_singular = [singular[i] for i in sorted_indices]
sorted_plural = [plural[i] for i in sorted_indices]
index_labels = [f"{c}-{p}" for c, p in zip(sorted_plural, sorted_singular)]

# creo df risultati
result_df = pd.DataFrame(matrix_sorted, index=index_labels, columns=sorted_singular)

# visualizzazione heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(result_df, cmap=sns.color_palette(["red", "blue"]), cbar=False, linewidths=0.5, linecolor='gray')
plt.title("Plural-Singular Animals Comparison Matrix with most_similar (Sorted)")
plt.xlabel("Singular")
plt.ylabel("Plural-Singular")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# statistiche
count_ones = np.sum(matrix == 1)
count_zeros = np.sum(matrix == 0)
total = count_ones + count_zeros
percent_ones = (count_ones / total) * 100

print(f"Conteggio '1': {count_ones}")
print(f"Conteggio '0': {count_zeros}")
print(f"Percentuale '1': {percent_ones:.2f}%")