import spacy

nlp = spacy.load("en_core_web_sm")

def summarize_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # 可以根据需要调整摘要的长度，这里简单地取前几个句子作为摘要
    summary = " ".join(sentences[:3])
    return summary

#text = "This is a long text. It contains many details. Another sentence. And one more."
#summary = summarize_text(text)
#print(summary)

file_path = "your_file.txt"
file_path = "labs.txt"
l = 0
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
    summary = summarize_text(text)
    print(f"Summary: {summary}")

exit()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        l = l +1 
        if l > 50:
            break
        summary = summarize_text(line.strip())
        print(f"Original Line: {line.strip()}")
        print(f"Summary: {summary}")
        print("-" * 30)
