import spacy
import re
nlp = spacy.load("en_core_web_sm")

log_text = "User with UID 1234 accessed path /home/user/file.txt from IP 192.168.1.100."

doc = nlp(log_text)

ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
path_pattern = r"/[a-zA-Z0-9_/.-]+|(?:[a-zA-Z]:\\[a-zA-Z0-9_\\/.-]+)"
uid_pattern = r"UID\s+\d+"

for ent in doc.ents:
    if re.match(ip_pattern, ent.text):
        print(f"IP Address: {ent.text}")
    elif re.match(path_pattern, ent.text):
        print(f"Path: {ent.text}")
    elif re.match(uid_pattern, ent.text):
        print(f"UID: {ent.text}")
    else:
        print(f"Other Entity: {ent.text}, Label: {ent.label_}")
