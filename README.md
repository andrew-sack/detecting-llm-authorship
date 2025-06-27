# Detecting LLM Authorship

Team Members: [Andrew Sack](https://github.com/andrew-sack), [Alireza Tehrani](https://github.com/Ali-Tehrani)

## Introduction
Large language models (LLM) have been a massive disruptive technology across many areas of fields, particularly academic, education and social media applications. 
Many users that utilize LLM can quickly synthesize large amounts of text geared to a specific purpose using prompts. They provide major productivity boost in many tasks, however
these texts can pose challenges in assessing students' ability to write, synthesizing unoriginal academic content, or using human bots to pose as humans on social networks.
This greatly warrants a business objective to decipher whether a given text is written in human form or LLM written. 

The ML objective of this project is to solve a binary classification problem whose input is a text file, and output is whether it is written by a LLM. The constraints are
to utilize classical (non-deep learning) machine learning techniques, and numerical descriptors of text files.

## Dataset
The AuTextification dataset is utilized for a competition from the 5th Workshop on Iberian Languages Evaluation Forum at the SEPLN 2023 Conference. 
This dataset provides 55677 short text-files either written by a human or written by a LLM. The prompt of the LLM is the initial prefix of the text and asked to auto complete the rest of the text.
The domains of the text are: legal, tweets, how to articles, news, and reviews.  The competition splits the dataset into training and test, where the tweets, legal and how to articles are 
within the training set, and the news and reviews are within the test set. 

<img src="https://github.com/user-attachments/assets/9366d669-07e8-4087-bc99-80967cfd2ec9" width=50% height=50%>

## Preprocessing

## Model Selection and Results

## File Descriptions

- [Data Folder](./Data/) : Folder that contains all of the datasets (both raw and clean).
