import pandas as pd

coca = pd.read_csv("COCA_WordFrequency.csv")
print(coca.head())

coca_sorted = coca.sort_values(by='freq', ascending=False)
print(coca_sorted.head())

lemmas = coca_sorted['lemma'].dropna().astype(str).str.lower()
lemmas.to_csv("coca_most_frequent_words.txt", index=False, header=False)
