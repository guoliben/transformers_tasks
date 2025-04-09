# from FlagEmbedding import FlagAutoModel
#
# model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
#                                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
#                                       use_fp16=True)
#
#
# sentences_1 = ["I love NLP", "I love machine learning"]
# sentences_2 = ["I love BGE", "I love text retrieval"]
# embeddings_1 = model.encode(sentences_1)
# embeddings_2 = model.encode(sentences_2)




from FlagEmbedding import FlagAutoReranker
pairs = [("样例数据-1", "样例数据-3"), ("样例数据-2", "样例数据-4")]
model = FlagAutoReranker.from_finetuned('BAAI/bge-reranker-large',
                                        use_fp16=True,
                                        devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
similarity = model.compute_score(pairs, normalize=True)
print(similarity)

pairs = [("query_1", "样例文档-1"), ("query_2", "样例文档-2")]
scores = model.compute_score(pairs)
print(scores)