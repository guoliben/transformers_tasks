# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# sequence = "测试是否支持中文分词--不支持"
# sequence = "This is not a good idea. YES OR NO!!!"
# sequence = "Using a Transformer network is simple"
# #['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '中', '文', '[UNK]', '[UNK]']
# tokens = tokenizer.tokenize(sequence)
#
#
#
# print(tokens)
#
#
# # --------------
# # print(AutoTokenizer.from_pretrained("bert-base-cased").tokenize("This is a good idea"))
#
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
#
# # --------------

#
#
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# sequence = "Using a Transformer network is simple"
# sequence_ids = tokenizer.encode(sequence)
#
# print(sequence_ids)

# # --------------

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# tokenized_text = tokenizer("Using a Transformer network is simple")
# print(tokenized_text)
#
# # # --------------

#
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
# print(decoded_string)
#
# decoded_string = tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102])
# print(decoded_string)
# # # # --------------
#
#
# from transformers import AutoTokenizer
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "So have I!"
# ]
# model_inputs = tokenizer(sequences)
# print(model_inputs)
#
# # # # --------------
#
# from transformers import AutoTokenizer
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
# print(tokenizer.tokenize(sentence))
# # # # --------------
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# num_added_toks = tokenizer.add_tokens(["new_token1", "my_new-token2"])
# print("We have added", num_added_toks, "tokens")
#
#
# new_tokens = ["new_token3", "my_new-token5"]
# new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
# tokenizer.add_tokens(list(new_tokens))
#
# print("We have added", new_tokens, "tokens")
#
#
# # # # --------------
# from transformers import AutoTokenizer, AutoModel
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
#
# print('vocabulary size:', len(tokenizer))
# num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
# print("After we add", num_added_toks, "tokens")
# print('vocabulary size:', len(tokenizer))
#
# model.resize_token_embeddings(len(tokenizer))
# print(model.embeddings.word_embeddings.weight.size())
#
# # Randomly generated matrix
# print(model.embeddings.word_embeddings.weight[-2:, :])


# # # # --------------

# 
# import torch
# 
# with torch.no_grad():
#     model.embeddings.word_embeddings.weight[-2:, :] = torch.zeros([2, model.config.hidden_size], requires_grad=True)
# print(model.embeddings.word_embeddings.weight[-2:, :])