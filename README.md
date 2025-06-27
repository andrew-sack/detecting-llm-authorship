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
This places more emphasis on the machine learning model to be generalizable, making it a much more challenging problem. Since the the test set has a different distribution compared to the training set, and is
outside of the interpolating regime.

<img src="https://github.com/user-attachments/assets/9366d669-07e8-4087-bc99-80967cfd2ec9" width=50% height=50%>

#### Word-Count
The total number of words varies between each domain. A histogram of the length of each text is presented below of the training and test set. It is incredibly difficult to compare sentences whose length is less than ten. 
In addition certain features, such as readability features, do not provide accurate values when the length of the word is less than 10. Therefore, we review data whose text is less than ten, resulting in final data set of N=33606 for training and N=21476 for test.  

<img src="https://github.com/user-attachments/assets/76b15380-67fa-4f61-8897-002f661d6d66" width=50% height=50%>

The following illustrates the word-count per domain of the training and test set. The tweet shows the lowest amount of word count, and all other domains shows high frequency of word-count between 60-80. 

<img src="https://github.com/user-attachments/assets/3b431ba3-a328-4948-b53c-c9a68efec3ed" width=75% height=75%>


## Feature Selection


## Model Selection and Results

## File Descriptions

- [Data Folder](./Data/) : Folder that contains all of the datasets (both raw and clean).
