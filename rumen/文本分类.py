from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"涉密公文应当标注份号，用于标识公文的印制顺序号。份号一般用6位阿拉伯数字表示，顶格编排在版心左上角第一行，例如“000001”。对于涉密文件，可以标注虚位，例如从“0001”号开始编码，一直到“0020”号。‌密级和保密期限‌：涉密公文应根据涉密程度分别标注“‌绝密”“‌机密”“‌秘密”三级，并用3号黑体字顶格编排在版心左上角第二行。保密期限中的数字用阿拉伯数字标注，例如“绝密★30年”“机密★20年”“秘密★10年",
# "您的工资情况本月到账10万元，未来中国经济发展会在5年内超越美国，中国建设银行投资南非钻石行业，美国大选开始了，合同编号12321",
# candidate_labels=["education", "politics", "business", "contract", "合同", "薪资"],)
candidate_labels=["用户手册","涉密文件", "部署文档", "技术分享"],)
print(result)

# 占用资源 内存475M CPU300%