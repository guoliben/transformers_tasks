from transformers import T5Tokenizer, T5ForConditionalGeneration  
  
# 加载预训练的T5模型和分词器  
model_name = "t5-small"  # 你可以使用 "t5-base", "t5-large", 或 "t5-3b" 等更大的模型  
tokenizer = T5Tokenizer.from_pretrained(model_name)  
model = T5ForConditionalGeneration.from_pretrained(model_name)  
  
# 定义输入文件和输出文件路径  
input_file_path = "input.txt"  
output_file_path = "output_summaries_t5.txt"  
  
# 读取输入文件内容  
with open(input_file_path, 'r', encoding='utf-8') as input_file:  
    lines = input_file.readlines()  

l=0
# 初始化输出文件  
with open(output_file_path, 'w', encoding='utf-8') as output_file:  
    for line in lines:  
        # 清理和预处理输入文本  
        text = line.strip()  
        l = l +1
        if l > 1:
            break
        # 如果文本为空，跳过  
        if not text:  
            continue  
          
        # T5模型需要特定的输入格式，即 "<prefix> " + text_to_summarize  
        # 这里我们使用 "summarize: " 作为前缀，但你可以根据需要更改  
        input_text = "summarize: " + text  
          
        # 编码输入文本  
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)  
          
        # 生成摘要  
        # 注意：T5模型在生成时默认会添加特定的输出标记（如 "</s>"），你可能需要在解码时处理这些标记  
        summary_ids = model.generate(inputs, max_length=150, min_length=30, num_return_sequences=1)  
          
        # 解码摘要  
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)  
          
        # 写入输出文件  
        output_file.write(summary + "\n")  
  
print("摘要生成完成，结果已保存到", output_file_path)
