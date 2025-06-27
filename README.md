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
The domains of the text are: legal, tweets, how to articles, news, and reviews.  

The competition splits the dataset into training and test, where the tweets, legal and how to articles are 
within the training set, and the news and reviews are within the test set.  This places more emphasis on the machine learning model to be generalizable, making it a much more challenging problem. Since the the test set has a different distribution compared to the training set, and is
outside of the interpolating regime.

<img src="https://github.com/user-attachments/assets/9366d669-07e8-4087-bc99-80967cfd2ec9" width=50% height=50%>

#### Word-Count
The total number of words varies between each domain. A histogram of the length of each text is presented below of the training and test set. It is incredibly difficult to compare sentences whose length is less than ten. 
In addition certain features, such as readability features, do not provide accurate values when the length of the word is less than 10. Therefore, we review data whose text is less than ten, resulting in final data set of N=33606 for training and N=21476 for test.  

<img src="https://github.com/user-attachments/assets/76b15380-67fa-4f61-8897-002f661d6d66" width=50% height=50%>

The following illustrates the word-count per domain of the training and test set. The tweet shows the lowest amount of word count, and all other domains shows high frequency of word-count between 60-80. 

<img src="https://github.com/user-attachments/assets/9b4e23fc-7635-408f-bbc5-bef03022a795" width=75% height=75%>

## Feature Selection

Different description of the text file is computed resulting in a total list of 316 features. We consider a wide variety of features ranging from simple features that include:

- Total amount of (unique) words
- Relative frequencies of different punctuation, and letters.
- The average length of words and sentences.
- Lexical diversity of the text, defined as the total number of unique words divided by the total number of words.
- The number of emotional words used, obtained from the(EmoLex) Word-Emotion Association Lexicon dataset.
- The number of common words used, obtained from the COCA Word Frequency dataset.

In addition, we employed some existing natural language processing libraries to extract features, including:

- The text's polarity (positive, neutral or negative feeling of the text), and its sentiment (personal opinion, and factual information)
- The grade-level or years of education needed to understand the text
- The relative frequency of vowel sounds.  

The following outlines our distribution of four of our features (average word length, number of words, average number of words and average sentence length) based on generated and human data:

  <img src="https://github.com/user-attachments/assets/a7b40468-ba66-4f04-b14d-13cb30501fe0" width=50% height=50%>


## Model Selection and Results

We utilized four very common machine learning mdoels to predict binary classification problems: random forest, Light Gradient Boosting Machine (Light-GBM), and feed-forward neural networks (FFNN). 
We remove 10/% of our training dataset as a hold-out set for comparing whether our model performs well on the non-interpolating regime from the initial test-set and from the interpolating regime. 


#### Random Forest
  We utilize a random forest from scikit-learn package.  All 316 features were utilized, and only a single hyper-parameter was utilized: the number of estimators. The value for the number of estimators was found to be 100.  The accuracies was found to be 0.79\% on the validation set and 64\% on the test set. The confusion plot of the validation is shown below, illustrating relatively the same level of false negatives and positives.
  
  <img src="https://github.com/user-attachments/assets/8763e686-5548-405d-a946-fbeb39d2915a" width=85% height=50%>

Whereas, the confusion plot of the test set is shown below,, illustrating much more false negatives than false positives.

  <img src="https://github.com/user-attachments/assets/dd3badaa-cbe2-49c5-b501-b3cbc64b6038" width=56% height=60%>

The Random Forest model identified these as the ten most influential features:

- Unique Word Count:                   0.035
- TTR Lexical Diversity:               0.027
- Hapax Rate:                0.023
- Sentence Length:                     0.017
- Average Sentence Length:             0.016
- Flesch reading score:                 0.015
- Gunning God Index:                   0.013
- Complex Word Frequencies:           0.012
- Frequencies of Proper Nouns:    0.012
- Average Word Length:                 0.012

#### FFNN

  We utilize a feedforward neural network from the scikit-learn package. All 316 features were utilized, and the hyper-parameters optimized were: number of hidden layers, activation function, optimization algorithm and choice of learning rate.  The accuracies was found to be 0.78 \% on the validation set and 62\% on the test set. Making it very similar to the random forest model. The following confusion matrices are shown of the validation and test set, respectively.  These demonstrate that the FFNN are able to have more balanced number of false negatives, and false positive.
  Since our business objective is based on fraud and spam detection, these indicate that the FFNN has much less false negatives, and would be more suited for these tasks.
  
  <img src="https://github.com/user-attachments/assets/536bfbab-6526-4490-9e4f-b81e94f7b49a" width=85% height=50%>

  <img src="https://github.com/user-attachments/assets/ff691adb-1edf-4ac4-9643-db33ed6b06f1" width=56% height=60%>


#### Light-GBM 


## File Descriptions

- [Data Folder](./Data/) : Folder that contains all of the datasets used (both raw, features and clean).
  - AuTextification dataset: contains the human and LLM written text used for classification. 
  - COCA Word Frequency dataset: contains the most frqeuent words from  the COCA (Corpus of Contemporary American English).
  - (EmoLex) Word-Emotion Association Lexicon dataset: contains a list of words that their basic emotion, and sentiment.
- [Deliverables Folder](./Deliberables) : Contains our business KPI report, and executive summary.
- [Scripts Folder](./Scripts): Contains all of our scripts used to generate, and produce these results.
