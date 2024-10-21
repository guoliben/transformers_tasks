from transformers import pipeline


# zh
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')

# -- zh end



ner = pipeline("ner", grouped_entities=True)
# ner = pipeline("text-classification", grouped_entities=True)
# results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
results = model("""Jhony is 35 years old. Lee has 10000 dollars;
以下是一些不同类型敏感数据的内容样例，但请注意，在实际应用中，必须严格保护敏感数据，避免泄露。

**一、个人身份信息**

1. 姓名：张三
2. 身份证号码：12345619900101XXXX
3. 电话号码：138XXXXXXXX
4. 地址：XX 省 XX 市 XX 区 XX 街道 XX 号

**二、财务信息**

1. 银行账号：1234567890123456
2. 信用卡号码：4111111111111111
3. 交易密码：XXXXXX（仅作为样例示意，实际中绝不能明文展示密码）
4. 支票号码：12345678
5. 投资组合详情：股票 A（100 股）、债券 B（$1000 面值）等

**三、医疗信息**

1. 病历号：20240904XXXX
2. 诊断结果：感冒/高血压/糖尿病等具体病症名称
3. 药物处方：阿莫西林，每次两粒，每日三次等
4. 体检报告数据：身高 175cm、体重 70kg、血压 120/80mmHg 等

**四、企业敏感信息**

1. 商业秘密配方：例如某种特殊饮料的成分比例（仅为示意，实际中商业秘密绝不能随意公开）
2. 客户名单：包含客户姓名、联系方式、购买历史等信息的表格片段，如“客户 A，电话 139XXXXXXXX，购买产品 X、Y”。
3. 财务报表摘要：收入 100 万元，利润 20 万元等。

再次强调，敏感数据的保护至关重要，以上内容仅为展示目的，在任何情况下都不能随意公开或泄露敏感数据。""")
print(results)
