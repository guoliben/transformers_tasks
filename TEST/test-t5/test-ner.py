from transformers import pipeline

# 加载一个预训练的中文NER模型
# 注意：这里使用的是'bert-base-chinese-ner'，但需要注意的是，这个模型ID可能不是官方的
# 你可能需要查找一个真正支持中文NER的模型，如'bert-base-chinese-finetuned-msra-ner'或其他
# 或者使用其他预训练模型，如ERNIE等
ner_pipeline = pipeline(
    "ner",
    model="bert-base-chinese-ner",  # 假设这是一个可用的中文NER模型
    tokenizer_name="bert-base-chinese",  # 如果模型与分词器不匹配，可以单独指定分词器
    grouped_entities=True
)

# 输入一段中文文本
text = "中国的首都是北京，它是一座历史悠久的城市。"

# 使用NER管道处理文本
results = ner_pipeline(text)

# 打印结果
for entity in results:
    print(entity)

# 输出可能类似于：
# {'word': '中国', 'entity': 'LOC', 'score': 0.99..., 'start': 0, 'end': 2}
# {'word': '北京', 'entity': 'LOC', 'score': 0.99..., 'start': 5, 'end': 7}
# ...