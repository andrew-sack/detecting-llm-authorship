# This module provides functions to extract various features from text.
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from textblob import TextBlob  # Useful for sentiment analysis
from typing import Union
from readability import Readability
import nltk
nltk.download('punkt_tab')
import warnings

# Add parsing the emotional and common words as global variables to speed up
with open("../Data/EmotionLexicon/unique_emotion_words.txt", "r") as f:
    emotional_words = set(line.strip().lower() for line in f)
with open("../Data/Common_Words/coca_most_frequent_words.txt", "r") as f:
    common_words = set(line.strip().lower() for line in f)


def split_words_remove_punctuation(text: str) -> list:
    r"""Splits a string and removes punctuation (e.g. capitalize, '?', '.', '!') from it."""
    return re.findall(r'\b\w+\b', text)

"""
Lexical Features
"""

def text_length(text: str) -> int:
    return len(text)

def word_count(text: str) -> int:
    return len(text.split())

def unique_word_count(text : str) -> int:
    # Make Apple, apple, apple!, apple? the same word
    return len(set(split_words_remove_punctuation(text)))

def average_word_length(text: str) -> float:
    # Make Apple, apple, apple!, apple? the same word
    words = split_words_remove_punctuation(text)
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

def ttr_lexical_diversity(text: str) -> float:
    r"""Type-Token Ratio: measures lexical diversity.

    Meaning comes from type = 'unique word', and token = 'any word'.
    Measures the ratio of unique words.
    """
    return unique_word_count(text) / word_count(text)

def hapax_legomenon_rate(text: str) -> float:
    r"""Hapax Legomenon Rate: Proportion of words that appear only once.

    Measure of the number of rare/unique words
    """
    count = dict({})
    words = split_words_remove_punctuation(text)
    total_words = len(words)
    # Avoid division by zero
    if total_words == 0:
        return 0.0
    for w in words:
        if w not in count:
            count[w] = 0.0
        count[w] += 1
    hapax_count = sum(1 for count in count.values() if count == 1)
    return hapax_count / total_words

"""
Syntactic Features
"""

def split_into_sentences(paragraph: str) -> list:
    # Regular expression to match sentence-ending punctuation
    # followed by a space and an uppercase letter (to reduce false splits)
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(paragraph.strip())
    return sentences

def sentence_length(text: str) -> int:
    sentences = split_into_sentences(text)
    return len(sentences)

def average_sentence_length(text: str) -> float:
    sentences = split_into_sentences(text)
    if len(sentences) == 0:
        return 0
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences)

def punctuation_frequencies(text: str) -> dict:
    # Count each punctuation mark including em dash
    punctuation_marks = r".,!?;:-—&_()\"'`"
    frequencies = {mark: text.count(mark) for mark in punctuation_marks}
    #normalize frequencies by total count of punctuation marks
    total_count = sum(frequencies.values())
    if total_count == 0:
        return {mark: 0 for mark in punctuation_marks}
    return {mark: count / total_count for mark, count in frequencies.items()}

def contraction_frequencies(text: str) -> float:
    # Obtain the frequencies of the contractions, don't, can't
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    if total_words == 0:
        return 0.0
    contractions = re.findall(r"\b\w+'\w+\b", text)
    return len(contractions) / total_words

def part_of_speech_frequencies(text: str):
    doc = nlp(text)
    pos_counts = {}
    #initialize counts for each part of speech
    # ADJ: adjective,
    # PRON: pronoun,
    # ADP: Adposition,  e.g. 'in' 'on' 'at' 'over'
    # AUX: Auxiliary,
    # CCONJ: stuff like and, but, or
    # DET: Stuff before a noun like 'this', 'a', 'those'
    # INTJ: Short exclamation, oh, wow, uh, ouch
    # NUM: Numerical words and numeric
    # PROPN: Proper nouns
    # SCONJ: 'because', 'although', 'if'
    # SYM: symbols '$', '?'
    for pos in [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
        "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "SCONJ", "SYM", "VERB", "X"
    ]:
        pos_counts[pos] = 0

    for token in doc:
        pos = token.pos_
        if pos not in pos_counts:
            continue
        pos_counts[pos] += 1
    #normalize frequencies by total count of tokens
    total_count = sum(pos_counts.values())
    if total_count == 0:
        return {pos: 0 for pos in pos_counts}
    return {pos: count / total_count for pos, count in pos_counts.items()}

def common_words_frequencies(text: str) -> float:
    # Gets the frequency of the most common words
    words = split_words_remove_punctuation(text)
    total = len(words)
    if total == 0:
        return 0.0
    common = sum(1 for word in words if word in common_words)
    return common / total

def complex_words_frequencies(text: str) -> float:
    # Gets the frequency of the most uncommon words
    words = split_words_remove_punctuation(text)
    total = len(words)
    if total == 0:
        return 0.0
    uncommon = sum(1 for word in words if word not in common_words)
    return uncommon / total

def complex_verb_count(text: str) -> float:
    doc = nlp(text)
    count = 0
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in common_words:
            count += 1
    return count


"""
Sentiment Features
"""
def emotional_word_frequencies(text: str) -> float:
    words = split_words_remove_punctuation(text)
    total = len(words)
    if total == 0:
        return 0.0
    emot_words = sum(1 for word in words if word not in emotional_words)
    return emot_words / total

def polarity_and_subjectivity(text: str) -> Union[float, float]:
    r"""Obtains the polarity and subjectivity

    The polarity is within [-1, 1] and measures the text’s sentiment orientation
     (positive, negative, or neutral)
    The sentiment is within [0, 1] and measures the amount of personal opinion and
    factual information in the text

    See https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
    """
    text_blob = TextBlob(text)
    return text_blob.sentiment

def vader_compound_index(text: str) -> float:
    raise NotImplementedError("Implement it probably using ntlk")


r"""
Readability
"""

def flesch_reading_ease(text: str) -> float:
    r"""
    Flesch Reading Ease: U.S. school-grade level required to understand the text.
    Value inbetween [0, 100]. But I'll normalize it to be within [0, 1].

    206.39  - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)

    """
    # Note that if it is less than 100 words it isn't as accurate
    if (word_count(text) < 100):
        warnings.warn("Less than 100 words means it is not accurate.")
    r = Readability(text, min_words=5)
    return r.flesch_kincaid().score / 100.0

def dale_chall_readability(text: str) -> float:
    r"""
     Grade level based on familiar vs. “hard” words plus sentence length.

     Percent of words not in 3,000 easy word list.

    """
    r = Readability(text, min_words=5)
    score = r.dale_chall().score
    return score

def gunning_gog_index(text: str) -> float:
    r = Readability(text, min_words=5)
    score = r.gunning_fog().score
    return score

def letter_frequencies(text: str) -> dict:
    """
    Calculate the frequency of each letter in the text.
    The function returns a dictionary with letters as keys and their frequencies as values.
    :param text: The input text to analyze.
    :return: A dictionary with letters as keys and their frequencies as values.
    """
    text = text.lower()
    frequencies = {chr(i): 0 for i in range(ord('a'), ord('z') + 1)}
    
    for char in text:
        if char.lower() in frequencies:
            frequencies[char] += 1
    
    total_count = sum(frequencies.values())
    if total_count == 0:
        return {char: 0 for char in frequencies}
    
    return {char: count / total_count for char, count in frequencies.items()}

import pronouncing
def extract_vowel_sounds(text: str) -> list:
    ARPAbet_vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", 
                      "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
    words = re.findall(r'\b\w+\b', text.lower())  # tokenize text
    vowels = []

    for word in words:
        phones_list = pronouncing.phones_for_word(word)
        if phones_list:
            # Use the first pronunciation
            phones = phones_list[0].split()
            for phone in phones:
                phoneme = re.sub(r'\d', '', phone)  # remove stress digits
                if phoneme in ARPAbet_vowels:
                    vowels.append(phoneme)
    
    return vowels

def vowel_sound_frequencies(text: str) -> dict:
    """
    Calculate the frequency of vowel sounds in the text.
    The function returns a dictionary with vowel sounds as keys and their frequencies as values.
    :param text: The input text to analyze.
    :return: A dictionary with vowel sounds as keys and their frequencies as values.
    """
    ARPAbet_vowels = {"AA", "AE", "AH", "AO", "AW", "AY", 
                      "EH", "ER", "EY", "IH", "IY", 
                      "OW", "OY", "UH", "UW"}
    vowels = extract_vowel_sounds(text)
    total_count = len(vowels)
    
    if total_count == 0:
        return {vowel: 0 for vowel in ARPAbet_vowels}
    
    frequencies = {vowel: 0 for vowel in ARPAbet_vowels}
    
    for vowel in vowels:
        frequencies[vowel] += 1
    
    return {vowel: count / total_count for vowel, count in frequencies.items()}

import itertools as it
def vowel_sound_pair_frequencies(text: str) -> dict:
    """
    Calculate the frequency of consecutive vowel sounds in the text.
    The function returns a dictionary with consecutive vowel sounds as keys and their frequencies as values.
    :param text: The input text to analyze.
    :return: A dictionary with consecutive vowel sounds as keys and their frequencies as values.
    """
    vowels = extract_vowel_sounds(text)
    ARPAbet_vowels = {"AA", "AE", "AH", "AO", "AW", "AY", 
                      "EH", "ER", "EY", "IH", "IY", 
                      "OW", "OY", "UH", "UW"}
    vowel_pairs = {sound:0 for sound in it.product(ARPAbet_vowels, repeat=2)}

    
    
    for i in range(len(vowels) - 1):
        vowel_pairs[(vowels[i], vowels[i + 1])] += 1
    total_count = sum(vowel_pairs.values())
    if total_count == 0:
        return  {"".join(pair):0 for pair in vowel_pairs}
    return {"".join(pair): count / total_count for pair, count in vowel_pairs.items()}