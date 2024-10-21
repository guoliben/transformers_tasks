# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# tokenizer.save_pretrained("./models/bert-base-cased/")
#



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")