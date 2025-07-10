from PIL import Image
from typing import List, Dict, Any, Optional, Callable
import torch
import time
import concurrent.futures
from transformers import BlipProcessor, BlipForQuestionAnswering
from utils.proprecss_llm_answer import extract_output, extract_answer
from KG_LLM.offline_process.kg import Node
from utils.utils import get_image_path
from KG_LLM.online_query.agent.base import baseAgent
from KG_LLM.online_query.agent.image.prompt import ImagePrompt
from KG_LLM.online_query.agent.image.ImageProcessor import ImageProcessor

class ImageAnalyzer(baseAgent):
    def __init__(self, base_config, kg_config):   
        super().__init__(base_config)
        self.kg_config = kg_config
        self.image_processor = ImageProcessor(base_config)
        self.prompt_generator = ImagePrompt()

    def inference(self, question: str, useful_information: str, image_entity_data_list: List[Node]):
        '''
        Image Agent Inference for ImageQA
    
        '''
        # Initialize the inference time counter
        inference_start_time = time.time()

        self.base_config.LLM.init_message()
        self.base_config.logger.log_info("----------agent:ImageAnalyzer被激活--------\n")
        
        # Step 1: Check if the question requires reasoning based on the objects in the image
        # self.base_config.logger.log_info("step 1:判断question是否需要借助图像的objects来确认条件,生成Does the image has...\n")
        # judge_object_prompt = self.prompt_generator.create_judge_object_prompt(question)
        # llm_result = self.base_config.LLM.predict(self.base_config.args, judge_object_prompt)
        # result = extract_output(llm_result)
        
        judge_object_question = ''
        image_has_object_entity_data_list = []  # List to store image data with objects
        image_object_information = {}  # Dictionary to store information about whether the image has objects
        image_question_list = []  # List to store image-related qa pairs
        image_object_flag = 0  # Flag to indicate if the image contains object information
        
        # if 'no'!= result.lower():
        #     image_object_flag = 1
        #     judge_object_question = result
        #     #Loop through image entity data list to indicate if the image contains object information
        #     for image_data in image_entity_data_list:
        #         image_title = image_data.title
        #         image_file = image_data.image_path
        #         image_answer = self.Image_qa(image_file, judge_object_question)
            
        #         if 'yes' in image_answer.lower():
        #             image_has_object_entity_data_list.append(image_data)
        #             image_object_information[image_title] = f'image title:{image_title} {judge_object_question}:{image_answer}\n' #TODO 是否需要把这个也记录到image_qa_list           
                       
        # else:
        #     image_has_object_entity_data_list = image_entity_data_list

        # self.base_config.logger.log_info(f"User prompt:{judge_object_prompt}\n")
        # self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
        # self.base_config.logger.log_info(f"Judge object question:{judge_object_question}\n")
        
        image_has_object_entity_data_list = image_entity_data_list
        # Step 2: Determine if the image is relevant to the question and proceed with further reasoning
        self.base_config.logger.log_info("step 2:判断图像是否与问题相关并生成与回答子问题\n")
        # Loop through all image data having the object
        for image_entity_data in image_has_object_entity_data_list:
            data_id = image_entity_data.data_id
            title = image_entity_data.title 
            

            alias_entity_data_list = [image_entity_data]
            #获取与图像同名同义的实体
            alias_entity_data_list += self.kg_config.KG.get_entity_by_entity(title, self.kg_config.data_id_list)
            #获取与图像异名同义的实体
            if (title, data_id) in self.kg_config.entity_link_hash.keys():
                alias_entity_data_list+=self.kg_config.entity_link_hash[(title, data_id)]
            alias_entity_data_list = list(set(alias_entity_data_list))

            if title in image_object_information.keys():
                reason_prompt = self.prompt_generator.create_pipeline_reason_prompt(question, useful_information, title, alias_entity_data_list, 1, judge_object_question)
            else:
                reason_prompt = self.prompt_generator.create_pipeline_reason_prompt(question, useful_information, title, alias_entity_data_list)
            
            self.base_config.LLM.init_message()
            llm_result = self.base_config.LLM.predict(self.base_config.args, reason_prompt)
            
            self.base_config.logger.log_info(f"User prompt:{reason_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{llm_result}\n")

            if 'no image' not in llm_result.lower():
                img_question, answer = self.image_processor.extract_question_and_answer(llm_result)
                if 'unknown' not in answer.lower():
                    image_question_list.append((data_id, title, img_question, answer))
                else:
                    answer = self.Image_qa(image_file, img_question)
                    image_question_list.append((data_id, title, img_question, answer))
                
                if image_object_flag == 1:
                    image_question_list.append((data_id, title, judge_object_question, 'yes'))
                
                for entity_data in alias_entity_data_list:
                    if entity_data.entity_name != title:
                        image_question_list.append((data_id, entity_data.entity_name, img_question, answer))
                        if image_object_flag == 1:
                            image_question_list.append((data_id, entity_data.entity_name, judge_object_question, 'yes'))
        self.base_config.logger.log_info(f'inference(image) time:{time.time()-inference_start_time}')
        return list(set(image_question_list))


    def Image_qa(self, image_path, question):
        processor = BlipProcessor.from_pretrained("/home/cmf/multiQA/loaded_models/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("/home/cmf/multiQA/loaded_models/blip-vqa-base")

        image = Image.open(image_path)
        inputs = processor(image, question, return_tensors="pt")

        generated_ids = model.generate(**inputs)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True)
        return answer



