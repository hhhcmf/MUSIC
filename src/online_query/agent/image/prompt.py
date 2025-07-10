from KG_LLM.online_query.agent.base import BasePromptGenerator
from data.preprocess.load_data import WEBQA_IMAGE_DATA
class ImagePrompt(BasePromptGenerator):
    def __init__(self, example=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example = example
    
    # def _get_information(self, image_title, alias_entity_data_list, image_object_flag, image_question):
    #     image_information = ''
    #     # for i, entity_data in enumerate(image_entity_data):
    #     pos = 0
    #     image_information += f'image title:{image_title}\ndescription:\n'
    #     for idx, entity_data in enumerate(alias_entity_data_list):
    #         description = entity_data.description
    #         image_information += f'{idx+1}.{description}\n'
    #         pos = idx+1
    #     if image_object_flag:
    #         image_information += f'{pos+1}.{image_question} answer:yes\n'
    #     return image_information
    
    def _get_information(self, image_title, entity_data, alias_information, image_object_flag, image_question, other_image_caption = ''):
        image_information = ''
        pos = 0
        image_description = ''
        description = entity_data.description.replace('\n', '').replace('\r', '')
        
        if (entity_data.title == image_title or entity_data.isImageTitle) and entity_data.type == 'image':
            if other_image_caption:
                image_description += f'the other information of the image:{other_image_caption}\n'
            image_description += f'image description:{description}\n'
            
        if image_object_flag:
            image_description += f'{image_question} answer:yes\n'

        image_information = f'image title:{image_title}\n{image_description}\n'
        if alias_information:
            image_information += f'the information of other modality data(这是其他模态的数据，不是图像模态的，但也是关于{image_title}这个实体):\n{alias_information}'
        return image_information

    def get_image_titles(self, image_entity_data_list):
        data_id_list = []
        image_information = ''
        for entity_data in image_entity_data_list:
            data_id = entity_data.data_id
            title = entity_data.title
            if data_id in data_id_list:
                continue
            else:
                data_id_list.append(data_id)
                other_image_caption = WEBQA_IMAGE_DATA[data_id].get('caption')
                image_information += f'data id:{data_id} title:{title} other information about the image:{other_image_caption}\n'
        return image_information


    def create_judge_object_prompt(self, question, *args, **kwargs):
        prompt = f'''
Please analyze the conditions mentioned in the question and determine which condition may require asking about the image for confirmation. If the image is needed to confirm the condition, please generate a question in the format "Does the image ...". If none of the conditions require asking about the image, please output "no".
Question:{question}
Please directly output the generated question or "no".Please output in the following format.
output:<the generated question> or 'no'

Examples:
example 1:
question:Which movie has the highest viewership: the one directed by Sani, or the one with a dog and a woman's face on the poster?
output:Does the image has a dog and a woman's face?
example 2:
question:What is the color of logo on the poster?
output:no
example 3:
question:What is the color of the flag of the country with the most medals at the Paris Olympics?
output:no
'''
        return prompt     

    def _create_judge_prompt(self, question, userful_information, image_title, alias_entity_data_list, image_object_flag = 0, image_question = '', *args, **kwargs):
        image_information = self._get_information(image_title, alias_entity_data_list, image_object_flag, image_question)
        prompt = f'''
You are a data analysis expert. I will provide a natural language question, known knowledge, and image informations. Please analyze the conditions mentioned in the question and determine which image meet the given criteria and directly output the image title.If none of the images are related to the question, respond with "no image."
Question:{question}
Known knowledge:\n{userful_information}
Image Informations:\n{image_information}
Please output in the following format.
Output:
"no image" or <image title>
..
'''
        return prompt

    def _create_reasoning_prompt(self, *args, **kwargs):
        prompt = '''
Next, analyze the image and generate one question to retrieve content from the image using the ImageQA API that will help answer the question effectively. Please output in the specified format.
API: ImageQA 
Description: This API is used for ask question to the image.The generated question should be concise and directly related to the image without any descriptive phrases, qualifiers, or conditions.
Input: {
    "title": "<The title of the image>",
    "question": "<The generated question>"
}  
Output: The answer to the image question.

Please output the API you will call. The output format is as follows(no any explanation, no ```json```):
{
    "API Access": {
        "API": "<Interface name>",
        "Input": <JSON-formatted input>
    }
}
        '''
        return prompt
    
#     def create_qa_prompt(self, question, image_title, alias_entity_data_list, image_object_flag = 0, image_question = '', *args, **kwargs):
#         image_information = self._get_information(image_title, alias_entity_data_list, image_object_flag, image_question)
#         prompt = f'''
# Given an image-related question and the information about the image, your task is to answer the image question based on this information. If you cannot answer the question based on the given image information, please output "unknown".
# image title:{image_title}
# image question:{question}
# information of the image:
# {image_information}

# Please output in the following format(If cannot answer the question based on the given image information, please output "unknown"):
# answer:<"unknown" or known answer>
#         '''
#         return prompt

#     def create_pipeline_reason_prompt(self, question, useful_information, image_title, alias_entity_data_list, image_object_flag = 0, image_question = '', *args, **kwargs):
#         image_information = self._get_information(image_title, alias_entity_data_list, image_object_flag, image_question)
#         prompt = f'''
# You are a data analysis expert. I will provide known knowledge, information about the image, and a natural language question. Your task is to analyze all the given information, determine whether the image is related to the question, and if it is, generate a question based on the image and answer the generated question with the provided information. This generated question should aim to extract visual information from the image that could be helpful in answering the original question.

# I will now provide the natural language question, known knowledge, and information about the image.
# Question:{question}
# Known knowledge:
# {useful_information}

# Information of the image:
# {image_information}

# To complete this task:
# First, you need to determine whether the given image is related to the provided natural language question based on the given information and image details. If it is not related, simply output "no image" without proceeding to step two and three.
# Second, if the image is relevant, generate a question based on the image. This generated question should aim to extract visual information from the image that is helpful for answering the original question (What/How...), or to determine whether the image contains the object specified in the original question (Did the image have...). And The generated question should be concise and directly related to the image without any descriptive phrases, qualifiers, or conditions. and output this generated image-question.
# Last, based on the provided knowledge and information, answer the generated image-question. If it is not possible to generate an answer from the given information, output "unknown."

# '''
#         prompt += '''
# Please output in the following format: (If the image is not related to the question, output "no image." If the image is related, output the generated question and the answer to that sub-question. If it is not possible to generate an answer based on the given information, the answer to the sub-question should be 'unknown.')
# "no image" or
# {
# generated question:<generated image-question>
# the answer about the generated question:<'unknown' or known answer>
# }   
#         '''
#         return prompt
    def create_pipeline_reason_prompt(self, question, useful_information, image_title, alias_entity_data_list, image_object_flag=0, image_question='', *args, **kwargs):
    
        image_information = self._get_information(image_title, alias_entity_data_list, image_object_flag, image_question)
        prompt = f'''
You are a data analysis expert. I will provide known knowledge, information about the image, and a natural language question. Your task is to analyze all the given information, determine whether the image is related to the question, and if it is, generate a question based on the image and answer the generated question using the provided information. This generated question should aim to extract visual information from the image that could be helpful in answering the original question.

I will now provide the natural language question, known knowledge, and information about the image.
Question: {question}
Known knowledge:
{useful_information}

Information of the image:
{image_information}

To complete this task:
1. First, determine whether the given image is related to the provided natural language question based on the given information and image details. If it is not related, simply output "no image" without proceeding to steps 2 and 3.
2. If the image is relevant, generate a question based on the image. This generated question should aim to extract visual information from the image that is helpful for answering the original question (e.g., What/How...), or to determine whether the image contains the object specified in the original question (Did the image have...). The generated question should be concise and directly related to the image, without any descriptive phrases, qualifiers, or conditions. Output this generated image-question.
3. Based on the provided knowledge and other modality data information (do NOT use image description information, as it is only a vague description and we have more reliable methods to obtain accurate image details), answer the generated image-question. If it is not possible to generate an answer from the given known knowledge and other modality data, output "unknown".
4. Finally, check whether the generated answer was based on the image description. If so, change the answer to "unknown", because image description information is not allowed to be used in this step.

Please output in the following format:
- If the image is not related to the question, output "no image".
- If the image is related, output the generated question and the answer to that sub-question.
- If the answer cannot be determined based on valid information, or if it was derived from the image description, output "unknown".
    '''
        prompt += '''
{
generated question:<generated image-question>
the answer about the generated question:<'unknown' or known answer>
}   
        '''
        return prompt

    def generate_question_with_information(self, question):
        prompt = f'''
The question is related to the image. Please analyze the given question and generate a sub-question to extract relevant information from the image. The generated sub-question does not need to include the title of the image or any other descriptive statements or conditions.
There are three types of questions:
The first type pertains to inquiries about visual information in the image, such as colors, scenes, or objects. For this type, generate a question in the format "What/How/Is... in the image?" to obtain visual information from the image to help answer the original question. Note that it should not be a general question like "What is in the image?" or any other type.
The second type involves locating an image that contains a specific object mentioned in the question (the object should be concrete, not a category or just the name of an object), without asking for specific details about the image. For this type, you should generate a question such as "Does the image have <some objects>?" to identify images that meet the conditions of the question.
Analyze which type the question belongs to and generate the corresponding sub-question.


Here are some examples(not specific task):
example 1:
question:Which movie has the highest viewership: the one directed by Sani, or the one with a dog and a woman's face on the poster?
output:Does the image have a dog and a woman's face?
example 2:
question:What is the color of the flag of the country with the most medals at the Paris Olympics?
output:What is the color of the flag in the image?
example 3:
question:What is the full title featuring a duck of in the Theater Credits of Michael Potts in 2002?
output:Does the image feature a duck?
example 4:
question:Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color?
output:What is the color in the image?
example 5:
question:Do the Northern Royal Flycatcher and the Blue-crowned Mot Mot have the same head shape?
output:What head shape in the image?
example 6:
question:Is the surface of the sculpture next to the bench at the art exhibition in Hyde Park, London smooth or rough?
output:Is the surface of the sculpture is smooth or rough?
example 7:
question:WHICH TWO COLORS ARE FOUND ON BOTH THE NORTHERN CARDINAL AND THE RED-FACED CORMORANT?
output:What is the color in the image?

specific task:
question:{question}

The generated sub-question does not need to include the title of the image or any other descriptive statements or conditions.
please output in following format:
output:<generated question>  
            '''
        return prompt
        
    def judge_object_prompt(self, question):
        prompt = f'''
Please analyze the conditions mentioned in the question and determine which condition may require asking about the image for confirmation. If the image is needed to confirm the condition, please generate a question in the format "Does the image have/feature <some object>",不要添加其他的描述性语句、条件等，. If none of the conditions require asking about the image, please output "no generated questions".

Here are some examples:
example 1:
question:Which movie has the highest viewership: the one directed by Sani, or the one with a dog and a woman's face on the poster?
output:Does the image have a dog and a woman's face?
example 2:
question:What is the color of the flag of the country with the most medals at the Paris Olympics?
output:no generated questions
example 3:
What is the full title featuring a duck of in the Theater Credits of Michael Potts in 2002?
output:Does the image feature a duck?
example 4:
question:In what book with exactly one person on their poster?
output:Does the image have exactly one person in the poster?
example 5:
question:Which bearded player was in the PTA team?
output:Does the image have a bearded person?
example 6:
question:What is the animal that has 1 position in the movie poster?
output:no generated questions

Specified task:
Question:{question}
Please directly output the generated question or "no generated questions".Please output in the following format.
output:<the generated question> or 'no generated questions'
            '''
        return prompt
        
    def judge_image_with_informtion(self, question, useful_information, image_title, image_entity_data, other_image_caption, alias_information, image_object_flag = 0, image_question = '', *args, **kwargs):
        image_information = self._get_information(image_title, image_entity_data, alias_information, image_object_flag, image_question, other_image_caption)
        print(image_information)
#         prompt = f'''
# Please determine whether the image information is relevant to the question based on the provided knowledge. Specify whether effective information needs to be derived from the image description or whether further details must be extracted from the image (due to insufficient information in the known data) or whether the presence of specific objects in the image must be confirmed to validate the question's conditions. 
# Note that text recognition in the image is not possible.

# Known knowledge:
# {useful_information}

# question:{question}
# Information of the image:
# {image_information}
# If image is related to the question or can generate the answer based on the given information or image information is relevant to the question or the question requires image-based verification (to obtain critical information not covered by the known data or to confirm the presence of specific objects in the image), output "yes". Otherwise, output "no". Provide the reason.

# Output format:
# reason:<reason>
# output:'yes'(yes indicates the the provided image {image_title} is related to question) or 'no' (no indicates the image is not related to the question)       
#         '''
        prompt = f'''
Please determine whether the image information is relevant to the question based on the provided knowledge. Specify whether effective information needs to be derived from the image description or whether further details must be extracted from the image (due to insufficient information in the known data) or whether the presence of specific objects in the image must be confirmed to validate the question's conditions. 
Note that text recognition in the image is not possible.

Known knowledge:
{useful_information}

question:{question}
Information of the image:
{image_information}
If image is related to the question or can generate the answer based on the given information or image information is relevant to the question or the question requires image-based verification (to obtain critical information not covered by the known data or to confirm the presence of specific objects in the image), output "yes". Otherwise, output "no". 

Output format:
output:'yes'(yes indicates the the provided image {image_title} is related to question) or 'no' (no indicates the image is not related to the question)       
        '''
        return prompt
    
    def judge_two_images(self, question, useful_image_entity_data_list):
        image_information = self.get_image_titles(useful_image_entity_data_list)
        prompt = f'''
Given that this question is related to images, determine whether the question requires comparing different images simultaneously. If required, identify which different images? If needed, output the corresponding data IDs of the images; otherwise, output "no".  
question:{question}
images:
{image_information} 

output format:
'no' or data id list:data id1,data id2,...
        '''
        return prompt

    def generate_entity_summary(self, alias_entity_data_list):
        alias_information, image_description = '', ''
        for idx, entity_data in enumerate(alias_entity_data_list):
            description = entity_data.description.replace('\n', '').replace('\r', '')
            alias_information += f'{idx+1}.{description}\n'
        prompt = f'''
Given various descriptions of the same entity, please summarize them into a summary.
{alias_information}
Output format:
output:<the summary of descriptions>
        '''
        return prompt
    

