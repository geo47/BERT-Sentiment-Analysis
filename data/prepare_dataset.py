import pandas as pd

from sklearn.model_selection import train_test_split

from data.preprocess import text_preprocessing_simple

TWO_CLASS_SENTIMENTS = ['negative', 'positive']
THREE_CLASS_SENTIMENTS = ['negative', 'neutral', 'positive']

DATASET_GOOGLE_PLAY = 'google_play'
DATASET_YELP = 'yelp'
DATASET_AIRLINE = 'airline'
DATASET_CoLA = 'cola'
DATASET_IMDB = 'imdb'


# def to_sentiment(rating):
#     rating = int(rating)
#     if rating == 1:
#         return 0
#     elif rating == 2:
#         return 1
#     elif rating == 3:
#         return 2
#     elif rating == 4:
#         return 3
#     else:
#         return 4


def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else:
    return 2


def create_dataset(dataset_type):
    if dataset_type == DATASET_AIRLINE:
        return create_2_class_dataset(dataset_type)
    elif dataset_type == DATASET_GOOGLE_PLAY or dataset_type == DATASET_YELP or \
        dataset_type == DATASET_IMDB:
        return create_3_class_dataset(dataset_type)


def create_2_class_dataset(dataset_type):

    df = pd.DataFrame()

    if dataset_type == DATASET_AIRLINE:
        df_complaint = pd.read_csv('data/' + dataset_type + '/complaint.csv')
        df_complaint['sentiment'] = 0
        df_non_complaint = pd.read_csv('data/' + dataset_type + '/non_complaint.csv')
        df_non_complaint['sentiment'] = 1

        df = df_complaint.append(df_non_complaint, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(['airline', 'id'], axis=1)
        df = df.rename(columns={'tweet': 'text'})
        df['text'] = df.text.apply(text_preprocessing_simple)
    elif dataset_type == DATASET_CoLA:
        df = pd.read_csv('data/' + dataset_type + '/in_domain_train.tsv', delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])
        df = pd.read_csv('data/' + dataset_type + '/in_domain_dev.tsv', delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])

        # Report the number of sentences.
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))

        # Display 10 random rows from the data.
        df.sample(10)

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=1000)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=1000)

    return df_train, df_test, df_val, TWO_CLASS_SENTIMENTS


def create_3_class_dataset(dataset_type):
    df = pd.read_csv('data/' + dataset_type + '/reviews.csv')

    if dataset_type == DATASET_GOOGLE_PLAY:
        df = df.rename(columns={'content': 'text'})
        df['sentiment'] = df.score.apply(to_sentiment)
    elif dataset_type == DATASET_YELP:
        df['sentiment'] = df.stars.apply(to_sentiment)
    elif dataset_type == DATASET_AIRLINE:
        df['sentiment'] = df.score.apply(to_sentiment)
    elif dataset_type == DATASET_IMDB:
        df['sentiment'] = df.score.apply(to_sentiment)

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=1000)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=1000)

    return df_train, df_test, df_val, THREE_CLASS_SENTIMENTS
