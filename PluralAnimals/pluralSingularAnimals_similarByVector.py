import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_path = "plural_singular_animals.csv"
df = pd.read_csv(input_path)
plural = df['plural'].tolist()
singular = df['singular'].tolist()

# matrice dei risultati
matrix = np.zeros((len(plural), len(singular)), dtype=int)

# loop su tutte le combinazioni: comp_i - pos_i + pos_j ≈ comp_j
for i in tqdm(range(len(plural)), desc="Processing rows"):
    comp_i = plural[i]
    pos_i = singular[i]

    try:
        comp_i_vec = word_vectors[comp_i]
        pos_i_vec = word_vectors[pos_i]

        for j in range(len(singular)):
            pos_j = singular[j]
            comp_j = plural[j]

            try:
                pos_j_vec = word_vectors[pos_j]
                target_vec = comp_i_vec - pos_i_vec + pos_j_vec

                predicted = word_vectors.similar_by_vector(target_vec, topn=1)[0][0]

                if predicted == comp_j:
                    matrix[i, j] = 1

            except KeyError:
                continue  # parola non presente

    except KeyError:
        continue  # parola non presente

# somma dei '1' per colonna
col_sums = matrix.sum(axis=0)

# ordina gli indici in base al numero di '1' in ordine decrescente
sorted_indices = np.argsort(-col_sums)

# riordina la matrice mantenendo la diagonale
matrix_sorted = matrix[np.ix_(sorted_indices, sorted_indices)]

# riordina le liste delle parole
sorted_singular = [singular[i] for i in sorted_indices]
sorted_plural = [plural[i] for i in sorted_indices]
index_labels = [f"{c}-{p}" for c, p in zip(sorted_plural, sorted_singular)]

# df dei risultati
result_df = pd.DataFrame(matrix_sorted, index=index_labels, columns=sorted_singular)

# heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(result_df, cmap=sns.color_palette(["red", "blue"]), cbar=False, linewidths=0.5, linecolor='gray')
plt.title("Plural-Singular Animals Comparison Matrix with similar_by_vector (Sorted)")
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
