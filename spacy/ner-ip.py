import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple is planning to open a new store in New York. Tim Cook is the CEO of Apple."

doc = nlp(text)

file_path = "your_file.txt"
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
    	text = line
	doc = nlp(text)
	

        for ent in doc.ents:
           print(f"Entity: {ent.text}, Label: {ent.label_}")
