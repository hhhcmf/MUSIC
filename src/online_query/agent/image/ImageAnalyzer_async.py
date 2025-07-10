from PIL import Image
from typing import List, Dict, Any, Optional, Callable
import torch
import time
import asyncio
import openai 
import base64
from collections import defaultdict
import os
from transformers import BlipProcessor, BlipForQuestionAnswering
from utils.proprecss_llm_answer import extract_output, extract_answer, extract_data_id
from KG_LLM.offline_process.kg import Node
from utils.utils import get_image_path
from KG_LLM.online_query.agent.base import baseAgent
from KG_LLM.online_query.agent.image.prompt import ImagePrompt
from KG_LLM.online_query.agent.image.ImageProcessor import ImageProcessor
from data.preprocess.load_data import WEBQA_IMAGE_DATA

class ImageAnalyzer(baseAgent):
    def __init__(self, base_config, kg_config):   
        super().__init__(base_config)
        self.kg_config = kg_config
        self.image_processor = ImageProcessor(base_config)
        self.prompt_generator = ImagePrompt()
        self.processor = BlipProcessor.from_pretrained("/home/cmf/multiQA/loaded_models/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("/home/cmf/multiQA/loaded_models/blip-vqa-base")
        self.qa_cache = {}
        self.qa_cache_dict = defaultdict(int)
        self.related_data_id_dict = defaultdict(int)
        self.related_data_cache = {}

    async def inference(self, question: str, useful_information: str, image_entity_data_list: List[Node], summarize_flag = 0):
        '''
        Image Agent Inference for ImageQA
    
        '''
        # Initialize the inference time counter
        inference_time = 0
        all_start_time = time.time()

        
        self.base_config.logger.log_info("----------agent:ImageAnalyzer被激活--------\n")
        # test = ''
        # for image_entity_data in image_entity_data_list:
        #     test += f'{image_entity_data.entity_id},{image_entity_data.entity_name}, {image_entity_data.isImageTitle}\n'
        # self.base_config.logger.log_info(f'image details:{test}')
        # print('图像个数：', len(image_entity_data_list))
        image_question_list = []
        judge_object_question = ''
        # if self.base_config.args.SHOW_LOGGER:
        #     self.base_config.logger.log_info("step 1:判断question是否需要借助图像的objects来确认条件,生成Does the image has...\n")
        # start_time = time.time()
        # self.base_config.image_llm.init_message()
        # judge_object_prompt = self.prompt_generator.judge_object_prompt(question)
        # llm_result, flag = self.base_config.LLM.predict(self.base_config.args, judge_object_prompt)
        # if not flag:
        #     raise ValueError('openai content error')

        # judge_object_question = extract_output(llm_result)
        # judge_object_question = judge_object_question.replace("very", "").strip()

        # if 'no generated questions' in judge_object_question.lower():
        #     judge_object_question = ''
        #     image_has_object_entity_data_list = image_entity_data_list
        #     image_object_information = {}
        #     image_object_flag = 0 
        # else:
        #     image_has_object_entity_data_list, image_object_information = await self.process_image_entities(question, image_entity_data_list, judge_object_question)
        #     if not image_has_object_entity_data_list:
        #         image_has_object_entity_data_list = image_entity_data_list
        #     image_object_flag = 1
        if self.base_config.args.SHOW_LOGGER:
            # self.base_config.logger.log_info(f"User prompt:{judge_object_prompt}\n")
            # self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
            # self.base_config.logger.log_info(f"Judge object question:{judge_object_question}\n")


            self.base_config.logger.log_info("step 2:判断图像是否与问题相关\n")
        
        tasks = []
        image_has_object_entity_data_list = image_entity_data_list
        judge_object_question = ''
        image_object_flag = 0
        image_object_information = {}
       
        cache_results = []
        for image_entity_data in image_has_object_entity_data_list:
            # result = await self.process_image_data(image_entity_data, question, useful_information, judge_object_question, image_object_flag, image_object_information, summarize_flag)
            # print('test', result)
            data_id = image_entity_data.data_id
            if self.related_data_id_dict[(data_id, image_entity_data.entity_name)]>2:
                self.base_config.logger.log_info('111111111')
                cache_results.append(self.related_data_cache[(data_id, image_entity_data.entity_name)])
                continue
            task = asyncio.create_task(self.process_image_data(image_entity_data, question, useful_information, judge_object_question, image_object_flag, image_object_information, summarize_flag))
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        image_questions, whole_titles, titles, data_ids = [], [], [], []
        image_paths = []
        useful_image_entity_data_list = []
        alias_names = {}
        data_id_path_dict = {}
        data_id_title_dict = {}

        results += cache_results
        for useful_image_entity_data, image_path, alias_name_list, title in results:
            if useful_image_entity_data:
                useful_image_entity_data_list.append(useful_image_entity_data)
                image_paths.append(image_path)
                titles.append(title)
                whole_titles.append(useful_image_entity_data.title)
                data_ids.append(useful_image_entity_data.data_id)
                alias_names[title] = alias_name_list
                data_id_path_dict[useful_image_entity_data.data_id] = image_path
                data_id_title_dict[useful_image_entity_data.data_id] = title
                self.base_config.logger.log_info(f"useful_image_entity_data:{useful_image_entity_data}\n")
        self.base_config.logger.log_info(f"useful_image_entity_data_list:{len(useful_image_entity_data_list)}\n")
        
        if useful_image_entity_data_list:
            if len(useful_image_entity_data_list) > 1:
                judge_images_prompt = self.prompt_generator.judge_two_images(question, useful_image_entity_data_list)
                self.base_config.image_llm.init_message()
                llm_result, flag = self.base_config.LLM.predict(self.base_config.args, judge_images_prompt)
                if not flag:
                    raise ValueError('openai content error')
                
                if self.base_config.args.SHOW_LOGGER:
                    self.base_config.logger.log_info(f"User prompt:{judge_images_prompt}\n")
                    self.base_config.logger.log_info(f"LLM response:{llm_result}\n")

                if 'no' not in llm_result:
                    useful_paths = []
                    image_data_ids = extract_data_id(llm_result)
                    image_data_id_list = image_data_ids.split(',')
                    title_list = []
                    for data_id in image_data_id_list :
                        if data_id in data_id_path_dict.keys():
                            useful_paths.append(data_id_path_dict[data_id])
                            title_list.append(data_id_title_dict[data_id])
                    if useful_paths:
                        answer = await self.images_qa_gpt(useful_paths, question)
                        alias_name_list = []
                        image_question_list.append((image_data_ids, ','.join(title_list).upper(), question.upper(), answer.upper(), tuple(alias_name_list)))
                        inference_time = time.time()-all_start_time
                        # print(f'all image time:{time.time()-all_start_time}')
                        return list(set(image_question_list)), useful_image_entity_data_list, judge_object_question, inference_time
           
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info("step 3:生成子问题\n")
            
            generate_prompt= self.prompt_generator.generate_question_with_information(question)
            self.base_config.image_llm.init_message()
            llm_result, flag = self.base_config.LLM.predict(self.base_config.args, generate_prompt)
            if not flag:
                raise ValueError('openai content error')

            generated_question = extract_output(llm_result).replace("very", "").strip()
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"User prompt:{generate_prompt}\n")
                self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
            self.base_config.logger.log_info(f"Generated question:{generated_question}\n")
            
            no_data_ids, no_titles, no_whole_titles = [], [], []
            no_image_paths, no_entity_data_list = [], []
            if image_paths:
                for idx, image_path in enumerate(image_paths):
                    if (image_path, titles[idx],  generated_question.lower()) in self.qa_cache.keys() and self.qa_cache_dict[(image_path, titles[idx], generated_question.lower())] > 2:
                        self.base_config.logger.log_info('22222222222')
                        continue
                    else:
                        no_data_ids.append(data_ids[idx])
                        no_titles.append(titles[idx])
                        no_whole_titles.append(whole_titles[idx])
                        no_image_paths.append(image_paths[idx]) 
                        no_entity_data_list.append(useful_image_entity_data_list[idx])

                start_time = time.time()
                image_questions = [generated_question] * len(no_image_paths)
                answers = await self.Image_qa(no_image_paths, image_questions, no_entity_data_list)
                for answer, img_question, data_id, title, whole_title in zip(answers, image_questions, no_data_ids, no_titles, no_whole_titles):
                    alias_name_list = alias_names[title]
                    if self.base_config.args.dataset == 'webqa':
                        other_image_caption = WEBQA_IMAGE_DATA[data_id].get('caption')
                        image_question_list.append((data_id, title.upper()+f'\nthe whole title of the image:{whole_title}\nother information of the image:{other_image_caption}', img_question.upper(), answer.upper(), tuple(alias_name_list)))
                    else:
                        image_question_list.append((data_id, title.upper(), img_question.upper(), answer.upper(), tuple(alias_name_list)))
                    if image_object_flag:
                        image_question_list.append((data_id, title.upper(), judge_object_question.upper(), 'yes', tuple(alias_name_list)))
                    self.base_config.logger.log_info(f"title:{title} the whole title of the image:{whole_title} {img_question}:{answer}\n")
        
        inference_time = time.time()-all_start_time
        # print(f'all image time:{time.time()-all_start_time}')
        return list(set(image_question_list)), useful_image_entity_data_list, judge_object_question, inference_time


    async def process_image_entities(self, question, image_entity_data_list, judge_object_question):
        """
        处理图像实体并使用批量推理
        """
        image_paths = [image_data.image_path for image_data in image_entity_data_list]
        questions = [judge_object_question] * len(image_paths)

        
        # 调用批量推理方法
        answers = await self.Image_qa(image_paths, questions)

        image_has_object_entity_data_list = []
        image_object_information = {}
        image_title_list = []
        for i, answer in enumerate(answers):
            # print(image_entity_data_list[i].data_id, image_entity_data_list[i].image_path, image_entity_data_list[i].description)
            if 'yes' in answer.lower() or image_entity_data_list[i].entity_name in question or image_entity_data_list[i].title in question:
                if image_entity_data_list[i].isImageTitle:
                    image_title = image_entity_data_list[i].entity_name
                else:
                    image_title = image_entity_data_list[i].title
                image_has_object_entity_data_list.append(image_entity_data_list[i])
                if image_title not in image_title_list:
                    image_title_list.append(image_title)
                    image_object_information[image_title] = f'image title:{image_title} {judge_object_question}:yes\n'
                self.base_config.logger.log_info(image_object_information[image_title])

        return image_has_object_entity_data_list, image_object_information

    # def get_related_images(self, image_entity_data_list, question, useful_information):
    #     image_information = ''
    #     for image_entity_data in image_entity_data_list:
    #         if image_entity_data.isImageTitle:
    #             title = image_entity_data.entity_name
    #         else:
    #             title = image_entity_data.title
    #         description = image_entity_data.description.replace('\n', ' ')
    #         data_id = image_entity_data.data_id
    #         image_information += f'image title:{title}\ndata id:{data_id}\nimage description:{description}\n\n'

    #     get_related_image_entity_prompt = self.prompt_generator.get_related_image_entity(image_information, question, useful_information)
    #     self.base_config.image_llm.init_message()
    #     llm_result, flag = await self.base_config.image_llm.predict_async(self.base_config.args, reason_prompt)
    #     if not flag:
    #         raise ValueError('openai content error')
    #     related_entities = extract_entity_and_data_id(llm_result)
    #     for entity_name, data_id in related_entities:
    #         useful_entity_data = self.kg_reason_processor.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
    #         image_path = image_entity_data.image_path
    #         if not os.path.exists(image_path):
    #             if self.base_config.args.dataset == 'webqa':
    #                 image_path = f'/data/cmf_dataset/WebQA/test_images/{image_path}'
    #         alias_name
    #         if useful_entity_data.isImageTitle:
    #             title = useful_entity_data.entity_name
    #         else:
    #             title = useful_entity_data.title
    #     if self.base_config.args.SHOW_LOGGER:
    #         self.base_config.logger.log_info(f"----get related images---\nUser prompt:{reason_prompt}\nLLM response:{llm_result}\n")

        
            
    
    async def process_image_data(self, image_entity_data, question, useful_information, judge_object_question, image_object_flag, image_object_information, summarize_flag):
        """
        处理图像数据并使用批量推理
        """

        other_image_caption = ''
        start_time = time.time()
        data_id = image_entity_data.data_id
        if image_entity_data.isImageTitle:
            title = image_entity_data.entity_name
        else:
            title = image_entity_data.title
        image_file = image_entity_data.image_path
        # if not os.path.exists(image_file):
        if self.base_config.args.dataset == 'webqa':
            image_id = ''.join([char for char in image_file if char.isdigit()])
            image_file = f'/data2/cmf/webqa/images/{image_id}.jpg'
        if self.base_config.args.dataset == 'webqa':
            # print('type', type(image_entity_data.data_id))
            # print(WEBQA_IMAGE_DATA)
            other_image_caption = WEBQA_IMAGE_DATA[image_entity_data.data_id].get('caption')
        # alias_entity_data_list = [image_entity_data]
        alias_entity_data_list = []
        # alias_entity_data_list += self.kg_config.KG.get_entity_by_entity(title, self.kg_config.data_id_list)
        
        # if (title, data_id) in self.kg_config.entity_link_hash.keys():
        #     alias_entity_data_list += self.kg_config.entity_link_hash[(title, data_id)]
        
        # alias_entity_data_list = list( set(alias_entity_data_list) - {image_entity_data} )

        alias_information = ''
        summarize_flag = False
        if summarize_flag and len(alias_entity_data_list)>1:
            generate_summary_prompt = self.prompt_generator.generate_entity_summary(alias_entity_data_list)
            self.base_config.image_llm.init_message()
            llm_result, flag = await self.base_config.image_llm.predict_async(self.base_config.args, generate_summary_prompt)
            alias_information = extract_output(llm_result)
            if not flag:
                raise ValueError('openai content error')
        
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"------entity summary--------\nUser prompt:{generate_summary_prompt}\nLLM response:{llm_result}\n")
        else:
            for idx, entity_data in enumerate(alias_entity_data_list):
                description = entity_data.description.replace('\n', '').replace('\r', '')
                alias_information += f'{idx+1}.{description}\n'

        if title in image_object_information.keys():
            reason_prompt = self.prompt_generator.judge_image_with_informtion(question, useful_information, title, image_entity_data, other_image_caption, alias_information, 1, judge_object_question)

        else:
            reason_prompt = self.prompt_generator.judge_image_with_informtion(question, useful_information, title, image_entity_data, other_image_caption, alias_information)
        
        self.base_config.image_llm.init_message()
        llm_result, flag = await self.base_config.image_llm.predict_async(self.base_config.args, reason_prompt)
        if not flag:
            raise ValueError('openai content error')
        
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{reason_prompt}\nLLM response:{llm_result}\n")
 

        image_path= None
        alias_name = []
        useful_image_entity_data = None

        if 'yes' in llm_result or title in question:
            image_path = image_entity_data.image_path
            # if not os.path.exists(image_path):
            if self.base_config.args.dataset == 'webqa':
                image_id = ''.join([char for char in image_path if char.isdigit()])
                image_path = f'/data2/cmf/webqa/images/{image_id}.jpg'

            useful_image_entity_data = image_entity_data
            for entity_data in alias_entity_data_list:
                if entity_data.entity_name != title:
                    alias_name.append(entity_data.entity_name)
        # print('test', useful_image_entity_data, image_path, alias_name, title)
        self.related_data_id_dict[(data_id, image_entity_data.entity_name)]+=1
        self.related_data_cache[(data_id, image_entity_data.entity_name)] = (useful_image_entity_data, image_path, alias_name, title)
        return useful_image_entity_data, image_path, alias_name, title

    async def Image_qa(self, image_paths: List[str], questions: List[str], entity_data_list):
        if self.base_config.args.engine.lower() == 'gpt-4o':
            tasks = []
            for image_path, question, entity_data in zip(image_paths, questions, entity_data_list):
                if not os.path.exists(image_path):
                    continue
                task = asyncio.create_task(self.image_qa_gpt(image_path, question, entity_data))
                tasks.append(task)
            answers = await asyncio.gather(*tasks)
        else:
            answers = await self.Image_qa_blip(image_paths, questions, entity_data_list)
        
        
        return answers


    async def Image_qa_blip(self, image_paths: List[str], questions: List[str], entity_data_list):
        """
        批量图像 QA 推理方法
        """
        
        # 批量加载图片
        # print(len(image_paths), image_paths)
        # print(len(questions), questions)
        images = [Image.open(image_path) for image_path in image_paths]

        # 批量处理输入
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True)
        
        # 推理
        generated_ids = self.model.generate(**inputs)
        
        # 解码所有生成的结果
        answers = [self.processor.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]

        for idx, answer in enumerate(answers):
            entity_data = entity_data_list[idx]
            if entity_data.isImageTitle:
                title = entity_data.entity_name
            else:
                title = entity_data.title
            image_path = image_paths[idx]
            self.qa_cache[(image_path, title, questions[0].lower())] = answer
            self.qa_cache_dict[(image_path, title, questions[0].lower())] += 1
        
        return answers

    async def image_qa_gpt(self, image_path, question, entity_data):
        openai.api_key = self.base_config.args.api_key[4]
        openai.api_base = self.base_config.args.api_base
        
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # 获取Base64字符串
        base64_image = encode_image(image_path)
        base_prompt = "Please answer concisely and directly, keep under 10 words."
        flag = False
        while not flag:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question+base_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                )
                flag = True
            except Exception as e:
                print(e)

        if entity_data.isImageTitle:
            title = entity_data.entity_name
        else:
            title = entity_data.title
        self.qa_cache[(image_path, title, question.lower())] = response.choices[0].message.content
        self.qa_cache_dict[(image_path, title, question.lower())] += 1
        return response.choices[0].message.content
    
    
    async def images_qa_gpt(self, image_paths, question):
        openai.api_key = self.base_config.args.api_key[4]
        openai.api_base = self.base_config.args.api_base

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_images = [encode_image(image_path) for image_path in image_paths]
        base64_images = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high",
            },
        }
        for base64_image in base64_images
        ]

        base_prompt = f"{question}.Please answer concisely and directly, keep under 10 words."
        user_content = [{"type": "text", "text": base_prompt}]
        user_content.extend(base64_images)
        messages_template = [{"role": "user", "content": user_content}]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages_template
        )

        return response.choices[0].message.content


