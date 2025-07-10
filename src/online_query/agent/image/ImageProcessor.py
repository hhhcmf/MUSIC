import os
import re
class ImageProcessor:
    def __init__(self, base_config):
        if base_config.args.dataset.lower() == 'mmqa':
            self.image_path = '/data/cmf/MMQA/data/final_dataset_images'
        elif base_config.args.dataset.lower() == 'manymodalqa':
            self.image_path = '/data/cmf_dataset/ManyModalQA/dev_images'
        elif base_config.args.dataset.lower() == 'webqa':
            self.image_path = '/data/cmf_dataset/WebQA/test_images'
        else:
            raise f"The TableProcessor doesn't support {base_config.args.dataset}"
    
    def get_image_path(self, image_id, img_folder):
        image_extensions = ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.bmp', '.tiff', '.gif', '.JPEG', '.GIF', '.BMP', '.TIFF', '.Jpg']
        for ext in image_extensions:
            potential_file = os.path.join(img_folder, f'{image_id}{ext}')
            if os.path.exists(potential_file):
                img_file = potential_file
                return img_file
        return None

    def extract_question_and_answer(self, text):
        pattern = r'["\']?generated question["\']?\s*[:|\s]\s*["\']?(.*?)["\']?\s*\n?["\']?the answer about the generated question["\']?\s*[:|\s]\s*["\']?(.*?)["\']?$'
    
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)

        if match:
            # 提取并清理问题部分和答案部分
            question = match.group(1).strip()
            answer = match.group(2).strip()
            return question, answer
        else:
            return None, None

# def extract_question_and_answer(text):
#     # 调整后的正则表达式模式，支持更灵活的格式匹配
#     pattern = r'["\']?generated question["\']?\s*[:|\s]\s*["\']?(.*?)["\']?\s*\n?["\']?the answer about the generated question["\']?\s*[:|\s]\s*["\']?(.*?)["\']?$'
    
#     match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)

#     if match:
#         # 提取并清理问题部分和答案部分
#         question = match.group(1).strip()
#         answer = match.group(2).strip()
#         return question, answer
#     else:
#         return None, None

# # 测试案例
# test_cases = [
#     '''generated question: "What color is the logo of Santa Anita Park?",
#     the answer about the generated question: "unknown"''',
#     """generated question: 'What is the capital of France?',
#     the answer about the generated question: 'Paris'""",
#     '''generated question: What year was the Eiffel Tower built?,
#     the answer about the generated question: 1889'''
# ]

# for test_str in test_cases:
#     q, a = extract_question_and_answer(test_str)
#     print(f"Question: {q}")
#     print(f"Answer: {a}")
#     print('-'*50)

# # def extract_question_and_answer(text):
# #     pattern = r'generated question:\s*(.*?)\nthe answer about the generated question:\s*(.*)'

# #     match = re.search(pattern, text, re.DOTALL)

# #     if match:
# #         question = match.group(1).strip()  # 提取问题部分
# #         answer = match.group(2).strip()
# #         return question, answer
# #     else:
# #         return None, None
# text = '''{
# "generated question": What is the title of the movie shown in the image?
# the answer about the generated question: The Hanging Tree
# }
# '''
# print(extract_question_and_answer(text))
    
    