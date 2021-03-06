# BERT-Sentiment-Analysis
Fine-Tuning BERT for Sentiment Analysis Task 
 


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

#### CoLA dataset
```
              precision    recall  f1-score   support

    negative       0.80      0.61      0.69       146
    positive       0.83      0.93      0.88       308

    accuracy                           0.83       454
   macro avg       0.82      0.77      0.79       454
weighted avg       0.82      0.83      0.82       454
```

#### smile-annotation dataset
```
              precision    recall  f1-score   support

       happy       0.88      0.95      0.91        56
not-relevant       0.64      0.50      0.56        14
       angry       0.00      0.00      0.00         2
     disgust       0.00      0.00      0.00         1
         sad       0.50      0.50      0.50         2
    surprise       0.00      0.00      0.00         0

    accuracy                           0.81        75
   macro avg       0.34      0.32      0.33        75
weighted avg       0.79      0.81      0.80        75
```

## References
- [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python]((https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/))

- [BERT Fine-Tuning Tutorial with PyTorch]((http://mccormickml.com/2019/07/22/BERT-fine-tuning/))