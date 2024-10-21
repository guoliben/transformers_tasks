import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("中国的首都是北京。")

for token in doc:
    print(token.text, token.pos_)

for ent in doc.ents:
    print(ent.text, ent.label_)
