import pandas as pd

COLUMNS = [
    "word",
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "negative",
    "positive",
    "sadness",
    "surprise",
    "trust"
]

records = []
with (open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r",encoding="utf-8")) as f:
    for line in f:
        word, emotion, value = line.strip().split('\t')
        if emotion in COLUMNS:  # only keep the 10 desired emotions
            records.append((word, emotion, int(value)))
df = (
    pd.DataFrame(records, columns=["word", "emotion", "value"])
    .pivot(index="word", columns="emotion", values="value")
    .fillna(0)                                      # missing → 0
    .astype(int)
)

print(df.head())


df.reset_index(inplace=True)

print(df.head())


# Save the full dataset
df.to_csv("emolex.csv")


# Only grab words that have atleast one emotion


# All columns representing emotions (excluding index and 'word')
emotion_columns = df.columns.difference(['word'])

# Select only rows where any emotion is 1
df_emotion = df[df[emotion_columns].sum(axis=1) > 0]

# Return set of words that are associated with at least one emotion
emotional_word_set = set(df_emotion['word'].str.lower())

with open("unique_emotion_words.txt", "w", encoding="utf-8") as f:
    for word in sorted(emotional_word_set):  # Optional: sort alphabetically
        f.write(word + "\n")
