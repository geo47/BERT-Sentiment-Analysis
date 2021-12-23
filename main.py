import time
from argparse import ArgumentParser
# from tqdm import tqdm

import random
import pandas as pd
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

from util.app_logging import desc_model
from util.util import format_time, evaluate_roc, make_confusion_matrix, make_training_roc

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

    #  Put the model into training mode. `train` just changes the mode, it doesn't
    #  perform the training. `dropout` and `batchnorm` layers behave differently
    #  during training vs. test
    #  (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model = model.train()

    losses = []
    correct_predictions = 0

    for step, batch in enumerate(data_loader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'
                  .format(step, len(data_loader), elapsed))

        # A `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())  # Accumulate the training loss over all of the batches

        loss.backward()

        # Clipping the norm of the gradients to 1.0 helps in preventing the "exploding gradients" problem.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()    # Update the learning rate.
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    # Prevent tp compute graph during the forward pass, since this is
    # only needed for backprop (training).
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
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--bert_model', type=str, default='bert-base-cased',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Total batch size for training. (BERT author recommendation: 16, 32)")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam. (BERT author recommendation: 5e-5, 3e-5, 2e-5)")
    parser.add_argument("--eps", default=1e-8, type=float,
                        help="epsilon parameter to prevent any division by zero in the implementation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--logging", action='store_true', help="Enable program level logging")
    args = parser.parse_args()

    # df_train, df_test, df_val = create_3_class_dataset(args.dataset)
    df_train, df_test, df_val, SENTIMENTS = create_dataset(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_data_loader = create_data_loader(df_train, tokenizer, args.max_len, args.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, args.max_len, args.batch_size)

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.logging:
        if is_cuda:
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')

    set_seed(42)

    # Initialize model
    model = BertSequentialSentimentClassifier(args.bert_model, len(SENTIMENTS))
    model = model.to(device)

    if args.logging:
        desc_model(model)

    total_steps = len(train_data_loader) * args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    total_t0 = time.time()

    # Todo add tqdm library
    for epoch in range(args.epochs):

        if args.logging:
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
            print('Training...')
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print("-" * 10)
        t0 = time.time()

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
        )

        train_time = format_time(time.time() - t0)
        if args.logging:
            print("")
            print(f"Epoch: {epoch}, Train loss: {train_loss}, accuracy: {train_acc}, elapsed time: {train_time}")

        if args.logging:
            print("")
            print("Running Validation...")

        t0 = time.time()

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, len(df_val)
        )

        val_time = format_time(time.time() - t0)
        if args.logging:
            print(f"Epoch: {epoch}, Val loss: {val_loss}, accuracy: {val_acc}, elapsed time: {val_time}")
        else:
            print(f"Epoch: {epoch}, Val loss: {val_loss}, accuracy: {val_acc}")

        history["epoch"].append(epoch + 1)
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["train_time"].append(train_time)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_time"].append(val_time)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), args.output_dir+'bert_senti_model.bin')
            best_accuracy = val_acc
    print("")
    print("Training complete!")

    if args.logging:
        print("Total training time {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        df = pd.DataFrame([history['epoch'],
                           torch.stack(history['train_acc']).cpu().numpy(),
                           history['train_loss'],
                           history['train_time'],
                           torch.stack(history['val_acc']).cpu().numpy(),
                           history['val_loss'],
                           history['val_time']])
        df = df.transpose()
        df.columns = ['epoch', 'train_acc', 'train_loss', 'train_time', 'val_acc', 'val_loss', 'val_time']
        print(df)

        make_training_roc(df, args.output_dir)

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    # evaluate_roc(y_pred_probs, y_test, args.output_dir)
    print("")
    print(classification_report(y_test, y_pred, target_names=SENTIMENTS))

    make_confusion_matrix(y_test, y_pred, SENTIMENTS, args.output_dir)
