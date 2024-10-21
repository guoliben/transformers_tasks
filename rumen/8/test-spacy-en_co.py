import spacy

nlp = spacy.load("en_core_web_sm")

text = "The company announced a new product launch."
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_, token.dep_, "------")

for ent in doc.ents:
    print(ent.text, ent.label_, "-------")


print(doc)
