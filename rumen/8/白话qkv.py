import pandas as pd
import numpy as np
import spacy

text1 = "我喜欢的水果是橙子和苹果, 我的名字张三，赵四 的手机号是15423212341, 欠了 100元，张三的社保账号12312312312,发送了1000个文件"
text2 = "相比苹果我更加喜欢国产的华为"

print("use spacy zh_core_web_sm model")
nlp = spacy.load('zh_core_web_sm')



doc = nlp(text1)



print(doc)
for token in doc:
	print("**************", token, end="：")
	print(token.text, end='___________')
	#print(token.vector, end='___________')
	print(token.lemma_, end='___________')
	print(token.lower_, end='___________')
	print(token.orth_, end='___________')
	print(token.pos_, end='___________')
	print(token.tag_, end='___________')
	print("[", token.dep_, end='_关系_]')
	print(token.head, end='___________')
	print("[", token.ent_type_, end='_ner_]')
	print("[", token.ent_iob_, end='_ner边界_]')
	print(token.is_alpha, end='___________')
	print(token.is_stop, end='___________')
	print(token.is_punct, end='___________')
	print(token.like_num, end='___________')
	print(token.is_oov, end='___________')
	print()



end_dim = 2

dics = {}
for token in doc:
    dics[token.text] = token.vector[:end_dim]


print(dics)
