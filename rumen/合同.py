from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-uncased-contracts")
model = AutoModel.from_pretrained("nlpaueb/bert-base-uncased-contracts")

# 新增


# 准备输入文本
text = "这里是你的合同文本示例，用于测试模型。"

# 使用tokenizer处理文本
# 注意：对于某些任务（如文本分类），我们可能需要添加特定的特殊标记，如[CLS]和[SEP]
# 这里我们简单使用encode_plus，它会自动添加这些标记（如果模型配置需要）
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 获取输入ID和注意力掩码
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 传递给模型
with torch.no_grad():  # 如果我们不需要计算梯度，可以加速计算
    outputs = model(input_ids, attention_mask=attention_mask)

# 处理输出
# 注意：模型的输出取决于它是如何被训练的。对于BERT，outputs通常包含last_hidden_state和pooler_output
# last_hidden_state是BERT编码器最后一层的输出，对于许多任务（如句子嵌入、文本相似性等）很有用
# pooler_output是[CLS]标记的输出，通常用于分类任务
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)
# 接下来，您可以根据需要对这些输出进行进一步的处理或分析
# 例如，提取[CLS]标记的嵌入用于分类任务，或计算两个文本嵌入之间的余弦相似度

# 如果这是一个分类任务，并且您已经有一个对应的分类头，您可能需要这样做：
# classifier_outputs = classifier(last_hidden_states[:, 0, :])  # 假设分类头只处理[CLS]标记的嵌入
classifier_outputs = model(last_hidden_states[:, 0, :])  # 假设分类头只处理[CLS]标记的嵌入
predicted_class = torch.argmax(classifier_outputs, dim=1)

# 注意：上述代码中的classifier_outputs和predicted_class部分需要您自己定义分类头
print(predicted_class)