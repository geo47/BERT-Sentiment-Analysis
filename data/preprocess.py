import nltk
import re

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# Uncomment to download "stopwords"
# nltk.download("stopwords")
from nltk.corpus import stopwords


def text_preprocessing_simple(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_preprocessing(s):
    """
    - In the bag-of-words model, a text is represented as the bag of its words,
      disregarding grammar and word order. Therefore, removing stop words,
      punctuations and characters that don't contribute much to the sentence's
      meaning will improve faster and better word representation.
    """

    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def tf_idf_vector(X_train, X_val):
    """
    - In information retrieval, TF-IDF, short for term frequency–inverse
      document frequency, is a numerical statistic that is intended to reflect
      how important a word is to a document in a collection or corpus.

      This function vectorize the text data using TF-IDF for feeding them to
      machine learning algorithms.

    :param X_train: training dataset
    :param X_val: validation dataset
    :return: vectorized training and validation dataset
    """
    # Preprocess text
    X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
    X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])

    # Calculate TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                             binary=True,
                             smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)