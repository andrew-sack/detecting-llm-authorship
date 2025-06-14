# This module provides functions to extract various features from text.

def text_length(text) -> int:
    return len(text)

def word_count(text) -> int:
    return len(text.split())

def average_word_length(text) -> float:
    words = text.split()
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


import spacy
nlp = spacy.load("en_core_web_sm")

def part_of_speech_frequencies(text: str):
    doc = nlp(text)
    pos_counts = {}
    #initialize counts for each part of speech
    for pos in ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]:
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



