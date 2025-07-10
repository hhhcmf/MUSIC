import os
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time
# 模型路径
model_path = "/loaded_models/blip-vqa-base"

# 加载模型和处理器
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForQuestionAnswering.from_pretrained(model_path)

# 图片文件路径列表
image_paths = [
]

# 问题
question = "What year was this movie released?"
image_path = '/ddf8b52a8400deaf05940c5cad8169cd.jpg'

# 加载图片
image = Image.open(image_path)

# 处理输入
start_time = time.time()
inputs = processor(image, question, return_tensors="pt")

# 推理
generated_ids = model.generate(**inputs)

# 解码输出
answer = processor.decode(generated_ids[0], skip_special_tokens=True)


# 输出结果
print(f"Answer: {answer}")
print(time.time()-start_time)

# 加载图片
images = [Image.open(image_path) for image_path in image_paths]

start_time = time.time()
# 处理批量输入
inputs = processor(images=images, text=[question] * len(images), return_tensors="pt", padding=True)

# 推理批量图片

generated_ids = model.generate(**inputs)

# 输出结果
for i, generated_id in enumerate(generated_ids):
    answer = processor.decode(generated_id, skip_special_tokens=True)
    print(f"Image {i + 1} Answer: {answer}")
print(time.time()-start_time)
