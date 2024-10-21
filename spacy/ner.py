import spacy
from spacy.tokens import Span
import re

nlp = spacy.load("en_core_web_sm")

# text = "Apple is planning to open a new store in New York. Tim Cook is the CEO of Apple."
# doc = nlp(text)



from spacy.matcher import PhraseMatcher


@spacy.Language.component("userid_ner")
def userid_ner(doc):
    # Create a PhraseMatcher object with the vocabulary from the doc
    matcher = PhraseMatcher(doc.vocab)
    # Tokenize the phrases in the POKEMON_NAMES list
    patterns = list(nlp.tokenizer.pipe(USER_IDS))
    # Add the patterns to the PhraseMatcher object
    matcher.add("USER_IDS", None, *patterns)
    # Find all matches in the doc using the PhraseMatcher object
    matches = matcher(doc)
    # Create a new Span object for each match
    spans = [Span(doc, start, end, label="USER_IDS") for match_id, start, end in matches]
    # Set the entities of the doc to the new spans
    doc.ents = spans
    # Return the updated doc
    return doc


# Define a list of Pokemon names
USER_IDS = ['uid', 'User id', 'USER_ID', 'USER ID', '1000']
# Create a blank spacy model and add the custom component to it
nlp = spacy.blank("en")
nlp.add_pipe("userid_ner", name="userid_ner")





# 定义一个函数来识别 IP 地址
@spacy.Language.component("ip_identifier")
def ip_identifier(doc):
    new_ents = []
    for ent in doc.ents:
        new_ents.append(ent)
    for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", doc.text):
        start, end = match.span()
        span = doc.char_span(start, end, label="IP_ADDRESS")
        if span is not None:
            new_ents.append(span)
    doc.ents = new_ents
    return doc

# 将自定义组件添加到管道中
nlp.add_pipe("ip_identifier", after="userid_ner")





@spacy.Language.component("uid_identifier")
def uid_identifier(doc):
    new_ents = []
    for ent in doc.ents:
        new_ents.append(ent)
    # for match in re.finditer(r"uid=\d+", doc.text):
    for match in re.finditer(r"(uid|user_id|UID)=\d+", doc.text):
        start, end = match.span()
        span = doc.char_span(start, end, label="UID")
        if span is not None:
            new_ents.append(span)
    doc.ents = new_ents
    return doc

# 将自定义组件添加到管道中
nlp.add_pipe("uid_identifier", after="ip_identifier")







@spacy.Language.component("file_path_identifier")
def file_path_identifier(doc):
   new_ents = []
   for ent in doc.ents:
      new_ents.append(ent)
   for match in re.finditer(r"(old_file=|new_file=|path=)([a-zA-Z0-9_/.-]+)", doc.text):
      start, end = match.span()
      span = doc.char_span(start, end, label="FILE_PATH")
      if span is not None:
         new_ents.append(span)
   doc.ents = new_ents
   return doc



nlp.add_pipe("file_path_identifier", after="ip_identifier")






@spacy.Language.component("comm_identifier")
def comm_identifier(doc):
   new_ents = []
   for ent in doc.ents:
      new_ents.append(ent)
   for match in re.finditer(r"(comm=)([a-zA-Z0-9_/.-]+)", doc.text):
      start, end = match.span()
      span = doc.char_span(start, end, label="COMM")
      if span is not None:
         new_ents.append(span)
   doc.ents = new_ents
   return doc



nlp.add_pipe("comm_identifier", after="file_path_identifier")






@spacy.Language.component("user_identifier")
def user_identifier(doc):
    new_ents = []
    for ent in doc.ents:
        new_ents.append(ent)
    for match in re.finditer(r"user\s+(\w+)", doc.text):
        start, end = match.span()
        span = doc.char_span(start, end, label="USER")
        if span is not None:
            new_ents.append(span)
    doc.ents = new_ents
    return doc

# 将自定义组件添加到管道中
nlp.add_pipe("user_identifier", after="comm_identifier")




@spacy.Language.component("time_identifier")
def time_identifier(doc):
    new_ents = []
    for ent in doc.ents:
        new_ents.append(ent)
    for match in re.finditer(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d+:\d+:\d+\b", doc.text):
        start, end = match.span()
        span = doc.char_span(start, end, label="TIME")
        if span is not None:
            new_ents.append(span)
    doc.ents = new_ents
    return doc

# 将自定义组件添加到管道中
nlp.add_pipe("time_identifier", after="comm_identifier")








# def summarize_line(line):
#     doc = nlp(line)
#     sentences = [sent.text for sent in doc.sents]
#     # 可以根据需要调整摘要的长度，这里简单地取第一个句子作为摘要
#     return sentences[0] if sentences else ""

# file_path = "your_file.txt"

# with open(file_path, "r", encoding="utf-8") as f:
#     for line in f:
#         summary = summarize_line(line.strip())
#         print(f"Original Line: {line.strip()}")
#         print(f"Summary: {summary}")
#         print("-" * 30)





nlp.add_pipe('sentencizer')



l = 0
file_path = "your_file.txt"
out_path = "output.txt"
with open(out_path, "w", encoding="utf-8") as out:
   with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
         print("----------------------------------------------")
      
         l = l + 1
         if l > 5000:
            break

         doc = nlp(line.strip())   
         sentences = [sent.text for sent in doc.sents]
         out.write(f"-----\nSummary")
         out.write(sentences[0] if sentences else "")


         for ent in doc.ents:
            print(f"{ent.label_} : {ent.text}")
            out.write(f"{ent.label_} : {ent.text}\n")

