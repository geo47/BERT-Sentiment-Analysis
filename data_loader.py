from torch.utils.data import Dataset, DataLoader
import torch


class PrepareDataset(Dataset):

    """
    @PrepareDataset Constructor

    @:param str[] reviews: list of reviews
    @:param int[] targets: list of review ratings
    @:param BERTTokenizer tokenizer:
    @:param int max_len: maximum sequence length
    """
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    """
    @:return dict{} -> review_text, [input_ids], [attention_mask], [targets]
    """
    def __getitem__(self, item_idx):
        review = str(self.reviews[item_idx])
        target = self.targets[item_idx]

        """
        For every review text, the `encode_plus` will:
        (1) Tokenize the sentence
        (2) Add the `[CLS]` and `[SEP]` token to the start and end
        (3) Truncate/Pad sentence to max length
        (4) Map tokens to their IDs
        (5) Create attention masks which explicitly differentiate real tokens from [PAD] tokens.
        (6) Return a dictionary of outputs
        """
        # `tokenizer.encode` returns above four features. However, `tokenizer.encode_plus` returns
        # all five features listed above.
        encoded_output = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.max_len,        # Max length to truncate/pad
            pad_to_max_length=True,         # Pad & truncate sentence to max length
            return_attention_mask=True,     # Return attention mask
            # return_token_type_ids=True,
            return_tensors="pt",            # Return PyTorch tensor
        )

        # uncomment if `return_tensors` is disabled
        # return {
        #     "review_text": review,
        #     "input_ids": torch.tensor(encoded_output["input_ids"]),
        #     "attention_mask": torch.tensor(encoded_output["attention_mask"]),
        #     "targets": torch.tensor(target, dtype=torch.long),
        # }

        # if `return_tensors` is enabled
        return {
            "review_text": review,
            "input_ids": encoded_output["input_ids"].flatten(),
            "attention_mask": encoded_output["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }
