import torch
import torch.nn as nn

from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f'Using {device} device')
print(torch.mps)




checkpoint = 'bert-base-cased'

class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        #super(BertForPairwiseCLSA, self).__init__()
        super().__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

model = BertForPairwiseCLS().to(device)
print(model)
