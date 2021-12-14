from transformers import BertModel
from torch import nn, optim


class SentimentClassifier(nn.Module):

    def __init__(self, bert_model, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        return self.out(output)
