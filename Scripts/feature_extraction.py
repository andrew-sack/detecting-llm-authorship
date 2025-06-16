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


def split_words_remove_punctuation(text: str) -> list:
    r"""Splits a string and removes punctuation (e.g. capitalize, '?', '.', '!') from it."""
    return re.findall(r'\b\w+\b', text)

def text_length(text) -> int:
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

import re

def split_into_sentences(paragraph) -> list:
    # Regular expression to match sentence-ending punctuation
    # followed by a space and an uppercase letter (to reduce false splits)
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(paragraph.strip())
    return sentences

def average_sentence_length(text: str) -> float:
    sentences = split_into_sentences(text)
    if len(sentences) == 0:
        return 0
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences)


def punctuation_frequencies(text: str):
    #Count each punctuation mark including em dash
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



