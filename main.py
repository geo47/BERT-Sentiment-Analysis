from argparse import ArgumentParser
from pathlib import Path
# from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel, get_linear_schedule_with_warmup,
)

from collections import defaultdict
from data_loader import PrepareDataset
from model import SentimentClassifier

import logging
logging.basicConfig(level=logging.ERROR)


# RATING_STARS = ['0', '1', '2', '3', '4']
RATING_STARS = ['negative', 'neutral', 'positive']


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

def create_data_loader(df, tokenizer, max_len, bs):
    ds = PrepareDataset(
        reviews=df["content"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=bs, num_workers=4)


def train_epoch(model, data_loader, batch_size, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions = correct_predictions + torch.sum(
            preds == targets
        )
        # print(type(correct_predictions.double()))
        # loss_batch = loss.item()
        # if (idx % print_every) == 0:
        #     print(f"The loss in {idx}th / {num_batches} batch is {loss_batch}")
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, tokenizer, text, max_len):
    model = model.eval()
    encoded_output = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    ids = encoded_output["input_ids"]
    masks = encoded_output["attention_mask"]
    outputs = model(input_ids=ids, attention_mask=masks)
    _, preds = torch.max(outputs, dim=1)
    probs = F.softmax(outputs, dim=1)
    return preds.numpy(), probs.detach().numpy()


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

        # predictions: [[], [], []]
    predictions = torch.stack(predictions).cpu()  # stack or concate lists of of tensors into single list of tensor
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default='data/google_play/reviews.csv')
    parser.add_argument('--output_dir', type=str, default='output/bert_senti_model.bin')
    parser.add_argument('--bert_model', type=str, default='bert-base-cased',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    args = parser.parse_args()


    df = pd.read_csv(args.dataset_file)
    df['sentiment'] = df.score.apply(to_sentiment)


    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_data_loader = create_data_loader(df_train, tokenizer, args.max_len, args.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, args.max_len, args.batch_size)

    model = SentimentClassifier(args.bert_model, len(RATING_STARS))
    model = model.to(device)

    total_steps = len(train_data_loader) * args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, args.batch_size, loss_fn, optimizer, device, scheduler, len(df_train)
        )

        print(f"Epoch: {epoch}, Train loss: {train_loss}, accuracy: {train_acc}")

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, len(df_val)
        )

        print(f"Epoch: {epoch}, Val loss: {val_loss}, accuracy: {val_acc}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model, args.output_dir)
            best_accuracy = val_acc

    # sample_text = "I really hate this movie"
    # predictions, probabilities = get_predictions(model, tokenizer, sample_text, args.max_len)
    # print(f'Text: {sample_text}, Prediction: {predictions[0]}, Probability: {np.max(probabilities[0])}')
    # print()

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )
    print(classification_report(y_test, y_pred, target_names=RATING_STARS))