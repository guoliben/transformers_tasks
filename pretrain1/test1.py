import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# 选择大模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备数据
data = pd.read_csv('data.csv')
print(data)
# 训练模型
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model.to(device)

class Trainer:
    def __init__(self, model, device, data):
        self.model = model
        self.device = device
        self.data = data

    def train(self, batch):
        # 训练逻辑
        inputs = tokenizer(batch['text'], return_tensors='pt')
        labels = torch.tensor(batch['label'])
        outputs = self.model(**inputs)
        loss = outputs.loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def evaluate(self, batch):
        # 评估逻辑
        inputs = tokenizer(batch['text'], return_tensors='pt')
        outputs = self.model(**inputs)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(-1) == labels).float().mean()
        return {'loss': loss.item(), 'accuracy': accuracy.item()}

# 构建知识库
knowledge_base = {}
for text in data['text']:
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    knowledge_base[text] = outputs.last_hidden_state[:, 0, :]

# 保存知识库
import pickle
with open('knowledge_base.pkl', 'wb') as f:
    pickle.dump(knowledge_base, f)

# 训练模型
trainer = Trainer(model, device, data)
batch_size = 32
for epoch in range(5):
    for i in range(len(data) // batch_size):
        batch = data.iloc[i * batch_size:(i + 1) * batch_size]
        trainer.train(batch)
    print(f'Epoch {epoch+1}, Loss: {trainer.evaluate(data.iloc[0:batch_size])[0]}')