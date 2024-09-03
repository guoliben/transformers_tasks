from transformers import pipeline

# 使用问答流水线
question_answerer = pipeline('question-answering', device="mps:0")
a  = question_answerer({
    'question': 'Transformers 框架原理?',
     'context': '''本文将从Transformer的本质、Transformer的原理、Transformer架构改进三个方面，带您一文搞懂Transformer。
图片

一、Transformer的本质
Transformer架构：主要由输入部分（输入输出嵌入与位置编码）、多层编码器、多层解码器以及输出部分（输出线性层与Softmax）四大部分组成。

图片

Transformer架构

输入部分：
源文本嵌入层：将源文本中的词汇数字表示转换为向量表示，捕捉词汇间的关系。
位置编码器：为输入序列的每个位置生成位置向量，以便模型能够理解序列中的位置信息。
目标文本嵌入层（在解码器中使用）：将目标文本中的词汇数字表示转换为向量表示。
编码器部分：
由N个编码器层堆叠而成。
每个编码器层由两个子层连接结构组成：第一个子层是一个多头自注意力子层，第二个子层是一个前馈全连接子层。每个子层后都接有一个规范化层和一个残差连接。
解码器部分：
由N个解码器层堆叠而成。
每个解码器层由三个子层连接结构组成：第一个子层是一个带掩码的多头自注意力子层，第二个子层是一个多头注意力子层（编码器到解码器），第三个子层是一个前馈全连接子层。每个子层后都接有一个规范化层和一个残差连接。
输出部分：
线性层：将解码器输出的向量转换为最终的输出维度。
Softmax层：将线性层的输出转换为概率分布，以便进行最终的预测。
Encoder-Decoder（编码器-解码器）：左边是N个编码器，右边是N个解码器，Transformer中的N为6。

图片

Encoder-Decoder（编码器-解码器）

Encoder编码器：
Transformer中的编码器部分一共6个相同的编码器层组成。
每个编码器层都有两个子层，即多头自注意力层(Multi-Head Attention)层和逐位置的前馈神经网络(Position-wise Feed-Forward Network)。在每个子层后面都有残差连接（图中的虚线）和层归一化（LayerNorm）操作，二者合起来称为Add&Norm操作。
图片

Encoder（编码器）架构

Decoder解码器：
Transformer中的解码器部分同样一共6个相同的解码器层组成。
每个解码器层都有三个子层，掩蔽自注意力层(Masked Self-Attention)、Encoder-Decoder注意力层、逐位置的前馈神经网络。同样，在每个子层后面都有残差连接（图中的虚线）和层归一化（LayerNorm）操作，二者合起来称为Add&Norm操作。
图片

Decoder（解码器）架构

二、Transformer的原理
图片

Transformer工作原理

Multi-Head Attention（多头注意力）：它允许模型同时关注来自不同位置的信息。通过分割原始的输入向量到多个头（head），每个头都能独立地学习不同的注意力权重，从而增强模型对输入序列中不同部分的关注能力。

图片

Multi-Head Attention（多头注意力）

输入线性变换：对于输入的Query（查询）、Key（键）和Value（值）向量，首先通过线性变换将它们映射到不同的子空间。这些线性变换的参数是模型需要学习的。
分割多头：经过线性变换后，Query、Key和Value向量被分割成多个头。每个头都会独立地进行注意力计算。
缩放点积注意力：在每个头内部，使用缩放点积注意力来计算Query和Key之间的注意力分数。这个分数决定了在生成输出时，模型应该关注Value向量的部分。
注意力权重应用：将计算出的注意力权重应用于Value向量，得到加权的中间输出。这个过程可以理解为根据注意力权重对输入信息进行筛选和聚焦。
拼接和线性变换：将所有头的加权输出拼接在一起，然后通过一个线性变换得到最终的Multi-Head Attention输出。'''
 })
# {'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

print(a)