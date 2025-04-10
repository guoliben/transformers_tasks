import requests
from PIL import Image
from transformers import pipeline

# Download an image with cute cats
# >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
# >>> image_data = requests.get(url, stream=True).raw
image = Image.open("300.jpeg")

# Allocate a pipeline for object detection
object_detector = pipeline(
    'object-detection',
    device="mps:0"
    )
#
# ner_pipeline = pipeline(
#     "ner",
#     model="bert-base-chinese-ner",  # 假设这是一个可用的中文NER模型
#     tokenizer_name="bert-base-chinese",  # 如果模型与分词器不匹配，可以单独指定分词器
#     grouped_entities=True
# )

object_detector(image)
# [{'score': 0.9982201457023621,
#   'label': 'remote',
#   'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
#  {'score': 0.9960021376609802,
#   'label': 'remote',
#   'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
#  {'score': 0.9954745173454285,
#   'label': 'couch',
#   'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
#  {'score': 0.9988006353378296,
#   'label': 'cat',
#   'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
#  {'score': 0.9986783862113953,
#   'label': 'cat',
#   'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]