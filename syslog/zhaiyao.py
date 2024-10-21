import torch  
from transformers import BartTokenizer, BartForConditionalGeneration  
  
# 加载预训练的BART模型和分词器  
model_name = "facebook/bart-large-cnn"  
tokenizer = BartTokenizer.from_pretrained(model_name)  
model = BartForConditionalGeneration.from_pretrained(model_name)  
  
# 定义输入文件和输出文件路径  
input_file_path = "/Users/ben/syslog-ana/dlp.log"  
output_file_path = "output_summaries.txt"  
  
# 打开输入文件读取内容  
with open(input_file_path, 'r', encoding='utf-8') as input_file:  
    lines = input_file.readlines()  

l=0
# 初始化输出文件  
with open(output_file_path, 'w', encoding='utf-8') as output_file:  
    for line in lines:  
        l = l +1
        if l > 1:
            break
        # 清理和预处理输入文本（去掉多余换行符等）  
        text = line.strip()  
          
        # 如果文本为空，跳过  
        if not text:  
            continue  
          
        # 编码输入文本  
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)  
          
        # 生成摘要  
        summary_ids = model.generate(inputs, num_beams=4, max_length=150, min_length=40, length_penalty=2.0, early_stopping=True)  
          
        # 解码摘要  
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)  
          
        # 写入输出文件  
        output_file.write(summary + "\n")  
  
print("摘要生成完成，结果已保存到", output_file_path)
