# NLP-Studies

This our repository that contains experiments on **linguistic analogies** using vector operations on word embeddings.  
The goal is to evaluate the performance of two different strategies:

**Most Similar (MS)**
This operation, finds the words most similar to a query vector while excluding the input words from the candidate set. This prevents trivial matches and ensures that the retrieved word is not simply one of the terms already used to build the query.

**Similar by Vector (SBV)**
This operation also finds the closest words to a query vector but does not exclude the input words. As a result, it may often return one of the original input terms if it is the closest vector.

These two methods provide complementary perspectives for evaluating how well vector offset operations capture semantic and morphological relationships between words.



## Project Structure

The repository is organized by **categories of word pairs**.  
Each folder contains:
- Python/Colab code for both `most_similar` and `similar_by_vector`
- The dataset of word pairs used in that category
- Generated plots 



The table below summarizes the correctness (percentage of correctly retrieved words) for each category, comparing MS and SBV.

| **Category**      | **Sample Word Pair** | **MS**   | **SBV**  |
|-------------------|----------------------|----------|----------|
| Capital Cities    | Norway – Oslo        | 78.78%   | 28.60%   |
| Currency          | Mexico – peso        | 0.75%    | 0.13%    |
| Man–Woman         | actor – actress      | 39.30%   | 10.40%   |
| Opposites         | hot – cold           | 12.08%   | 2.00%    |
| Comparatives      | tall – taller        | 81.35%   | 11.42%   |
| Verb Forms        | work – works         | 71.44%   | 13.06%   |
| Plural Nouns      | chair – chairs       | 67.80%   | 5.21%    |
| Plural Animals    | cat – cats           | 83.14%   | 9.44%    |
| Plurals           | box – boxes          | 59.38%   | 4.98%    |


## References

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013a).  
  *Efficient Estimation of Word Representations in Vector Space.*  
  arXiv preprint arXiv:1301.3781. [Link](https://arxiv.org/abs/1301.3781)

- Mikolov, T., Yih, W.-t., & Zweig, G. (2013b).  
  *Linguistic Regularities in Continuous Space Word Representations.*  
  In Vanderwende, L., Daumé III, H., & Kirchhoff, K. (eds.), *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, pp. 746–751.  
  Association for Computational Linguistics. [Link](https://aclanthology.org/N13-1090)
