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
        return {pair: 0 for pair in vowel_pairs}
    return {"".join(pair): count / total_count for pair, count in vowel_pairs.items()}