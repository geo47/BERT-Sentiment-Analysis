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
        encoded_output = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoded_output["input_ids"].flatten(),
            "attention_mask": encoded_output["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }
