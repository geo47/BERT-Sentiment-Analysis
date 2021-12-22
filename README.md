# BERT-Sentiment-Analysis
Fine-Tuning BERT for Sentiment Analysis Task 

Reference: [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python]((https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)) 


## Test Results

#### GooglePlay dataset
```
              precision    recall  f1-score   support

    negative       0.93      0.88      0.90       245
     neutral       0.84      0.88      0.86       254
    positive       0.92      0.92      0.92       289

    accuracy                           0.89       788
   macro avg       0.90      0.89      0.89       788
weighted avg       0.90      0.89      0.90       788
```

#### Yelp dataset
```
              precision    recall  f1-score   support

    negative       0.85      0.82      0.84       329
     neutral       0.49      0.54      0.51       228
    positive       0.81      0.80      0.80       443

    accuracy                           0.74      1000
   macro avg       0.72      0.72      0.72      1000
weighted avg       0.75      0.74      0.75      1000
```

#### Airline tweets dataset
```
              precision    recall  f1-score   support

    negative       0.76      0.86      0.81        88
    positive       0.83      0.71      0.76        82

    accuracy                           0.79       170
   macro avg       0.79      0.79      0.79       170
weighted avg       0.79      0.79      0.79       170
```