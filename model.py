from transformers import BertModel
from torch import nn, optim


class BertSentimentClassifier(nn.Module):

    def __init__(self, bert_model, num_classes):
        """
        @param    bert: a BertModel object
        @param    num_classes: number of target labels
        """
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Feed input to the model to compute output.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information
                  with shape (batch_size, max_length)
        @return   out (logits) (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        return self.out(output)


class BertSequentialSentimentClassifier(nn.Module):

    def __init__(self, bert_model, num_classes, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    num_classes: number of target labels
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertSequentialSentimentClassifier, self).__init__()
        LINEAR_HIDDEN_SIZE = 50
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, LINEAR_HIDDEN_SIZE),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(LINEAR_HIDDEN_SIZE, num_classes)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to the model to compute output.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information
                  with shape (batch_size, max_length)
        @return   out (logits) (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = pooled_output[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        return logits
