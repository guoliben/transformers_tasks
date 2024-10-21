from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 加载预训练的模型和分词器
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 假设是二分类问题

# 加载数据集（这里以虚构的数据集为例，实际中你需要加载自己的数据集）
# 注意：Transformers库与datasets库集成得很好，但这里为了简化示例，我们直接构造一些数据
from datasets import Dataset

texts = ["我喜欢这个产品", "这个产品太差了", "服务很好", "服务很糟糕"]
labels = [1, 0, 1, 0]  # 假设1表示正面评价，0表示负面评价

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})


# 数据预处理函数，将文本转换为模型可以接受的格式
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 输出文件夹
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=16,  # 每个设备的批处理大小
    per_device_eval_batch_size=64,  # 评估时的批处理大小
    warmup_steps=500,  # 预热步数
    weight_decay=0.01,  # 权重衰减
    logging_dir='./logs',  # 日志文件夹
    logging_steps=10,
)

# 初始化Trainer
trainer = Trainer(
    model=model,  # 加载的模型
    args=training_args,  # 训练参数
    train_dataset=tokenized_datasets,  # 训练数据集
    # eval_dataset=tokenized_datasets['test']  # 如果有测试集，可以取消注释并传入
)

# 开始训练
trainer.train()

# 注意：上面的代码示例中，我们直接使用了虚构的数据集，并且没有划分训练集和测试集。
# 在实际应用中，你需要加载自己的数据集，并进行适当的划分和预处理。

# 如果你想要对新的文本进行分类，可以使用以下代码：
# inputs = tokenizer("这是一个新的文本", return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs, labels=None)
# logits = outputs.logits
# # 假设是二分类问题，取logits的第一个维度的最大值对应的索引作为预测结果
# predicted_class_idx = torch.argmax(logits, dim=1).item()
# print(f"Predicted class: {predicted_class_idx}")