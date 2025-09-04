# NLP Analysis of Energy Data

This project demonstrates Natural Language Processing (NLP) techniques applied to the Our World in Data (OWID) energy dataset. Since the dataset primarily contains numerical data, the analysis creatively focuses on available textual elements and includes a synthetic text demonstration.

## Project Structure

The analysis consists of two main parts:

### Part 1: Country Name Clustering
- **Objective:** Cluster countries based on linguistic similarities in their names
- **Techniques:** Character-level TF-IDF vectorization (n-grams 2-4) + K-Means clustering
- **Output:** Groups of countries with similar name patterns (e.g., similar suffixes, phonetic structures)

### Part 2: Synthetic Energy Topic Modeling
- **Objective:** Demonstrate standard NLP topic modeling workflow
- **Techniques:** Text preprocessing, TF-IDF vectorization, Non-Negative Matrix Factorization (NMF)
- **Output:** Identified topics in energy-related synthetic text (renewables, traditional energy, supporting technologies)

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
# NLP-Project
