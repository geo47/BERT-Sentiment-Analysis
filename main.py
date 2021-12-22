from argparse import ArgumentParser
# from tqdm import tqdm

import random
import numpy as np

from sklearn.metrics import classification_report

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel, get_linear_schedule_with_warmup,
)

from collections import defaultdict
from data.prepare_dataset import create_dataset
from data_loader import PrepareDataset
from model import BertSentimentClassifier, BertSequentialSentimentClassifier

import logging
logging.basicConfig(level=logging.ERROR)


def create_data_loader(df, tokenizer, max_len, bs):
    ds = PrepareDataset(
        reviews=df["text"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=bs, num_workers=4)


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
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
        correct_predictions += torch.sum(preds == targets)
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

        # predictions: [[], [], []]
    predictions = torch.stack(predictions).cpu()  # stack or concat lists of of tensors into single list of tensor
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='google_play')
    parser.add_argument('--output_dir', type=str, default='output/bert_senti_model.bin')
    parser.add_argument('--bert_model', type=str, default='bert-base-cased',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    args = parser.parse_args()

    # df_train, df_test, df_val = create_3_class_dataset(args.dataset)
    df_train, df_test, df_val, SENTIMENTS = create_dataset(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_data_loader = create_data_loader(df_train, tokenizer, args.max_len, args.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, args.max_len, args.batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    set_seed(42)

    # Initialize model
    model = BertSequentialSentimentClassifier(args.bert_model, len(SENTIMENTS))
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

    # Todo add tqdm library
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
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
            torch.save(model.state_dict(), args.output_dir)
            best_accuracy = val_acc

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )
    print(classification_report(y_test, y_pred, target_names=SENTIMENTS))