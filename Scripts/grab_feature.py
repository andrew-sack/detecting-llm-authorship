import pandas as pd
from feature_extraction import *
import numpy as np

TASK = "one"
df = pd.read_csv(
    f"../Data/AuTextification/cleaned/subtask_{TASK}_grouped.tsv",
    sep="\t",
    header=0,
    index_col=0
)

features_arr = np.empty((len(df), 316), dtype=float)
for index, row in df.iterrows():
    text = row["text"]
    f1 = text_length(text)
    f2 = word_count(text)
    f3 = unique_word_count(text)
    f4 = average_word_length(text)
    ttr = ttr_lexical_diversity(text)
    hepa = hapax_legomenon_rate(text)
    f7 = sentence_length(text)
    f8 = average_sentence_length(text)
    freq_cont = contraction_frequencies(text)
    keys = ['.', ',', '!', '?', ';', ':', '-', '—', '&', '_', '(', ')', '\\', '"', "'", '`']
    punctuation = punctuation_frequencies(text)
    speech = part_of_speech_frequencies(text)
    keys2 = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
             'PART', 'PRON', 'PROPN', 'SCONJ', 'SYM', 'VERB', 'X']
    com_words = common_words_frequencies(text)
    comp_words = complex_words_frequencies(text)
    comp_verb = complex_verb_count(text)
    emot = emotional_word_frequencies(text)
    polarity, sentiment = polarity_and_subjectivity(text)
    flesch = flesch_reading_ease(text)
    dale = dale_chall_readability(text)
    gog = gunning_gog_index(text)

    # Vowels features
    letter_freq = letter_frequencies(text)
    vowel_sound_freq = vowel_sound_frequencies(text)
    vowel_pair_freq = vowel_sound_pair_frequencies(text)

    # Should match the ordering
    COLUMNS = [
        "text_length", "word_count", "unique_word_count", "average_word_length",
        "ttr_lexical_diversity", "hapax_rate", "sentence_length", "average_sentence_length",
        "contraction_freq"
    ]
    COLUMNS += [f"punctuation_freq_{k}" for k in keys]
    COLUMNS += [f"part_of_speech_freq_{k}" for k in keys2]
    COLUMNS += ["common_words_freq", "complex_word_freq", "complex_verb_count",
                "emotion_word_freq", "polarity", "sentiment", "flesh_readability",
                "dale_readability", "gunning_gog_index"]

    features = [
        f1, f2, f3, f4, ttr, hepa, f7, f8, freq_cont
    ]
    features += [punctuation[k] for k in keys]
    features += [speech[k] for k in keys2]
    features += [com_words, comp_words, comp_verb, emot, polarity, sentiment, flesch,
                 dale, gog]
    print(index / len(df)  *100, features)

    for freq_dict in [letter_freq, vowel_sound_freq, vowel_pair_freq]:
        for name, feature in freq_dict.items():
            COLUMNS.append(name)
            features.append(feature)

    assert len(features) == len(COLUMNS)

    features_arr[index] = features


np.save(f"../Data/AuTextification/features/full_features_subtask_{TASK}.npy", features_arr)