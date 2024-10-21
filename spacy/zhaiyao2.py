import spacy

nlp = spacy.load("en_core_web_sm")

def summarize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    doc = nlp(text)
    summary = []
    for sent in doc.sents:
        if len(summary) < 30:  # 可以根据需要调整摘要句子数量
            summary.append(sent.text)
    return " ".join(summary)

file_path = "labs.txt"  # 替换为实际的文件路径
print(summarize_file(file_path))
