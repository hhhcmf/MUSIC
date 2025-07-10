from online_query.agent.base import baseAgent
from online_query.agent.text.prompt import TextPrompt
from online_query.agent.text.TextProcessor import TextProcessor
from utils.proprecss_llm_answer import extract_output
import time
import asyncio

class TextExtractor(baseAgent):
    def __init__(self, base_config, kg_config):   
        super().__init__(base_config)
        self.kg_config = kg_config
        self.prompt_generator = TextPrompt(kg_config)
        self.text_processor = TextProcessor()
        # self.retrieve_entities_phase = retrieve_entities_phase

    async def inference(self, question, userful_information, text_entity_data_list):
        '''
        针对文本模态的推理,返回与question有关的文本实体信息,并且包含相关的文本片段,组成
        '''
        all_start_time = time.time()
        self.base_config.logger.log_info("----------agent:TextExtractor被激活--------\n")
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info("step 1:获取与question有关的文本实体\n")
        #step 1:获取与question有关的实体
        def chunkify(lst, chunk_size=6):
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        chunks = list(chunkify(text_entity_data_list, 6))

       
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(self.ExtractTextEntities(question, userful_information, chunk))

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # self.base_config.logger.log_info(f"LLM 执行提取实体 time：{time.time()-start_time}")

        related_entities = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 如果该批次出现异常，可自行处理或跳过
                self.base_config.logger.log_info(f"LLM 执行批次 {i} 发生异常：{result}")
                continue
            related_entity, inference_time = result
            # self.base_config.logger.log_info(f"LLM 执行批次 {i} time：{inference_time}")
            related_entities.extend(related_entity)

        # 若没有找到任何实体，直接返回
        if not related_entities:
            inference_time = time.time() - start_time
            # print(f'all text time:{time.time()-all_start_time}')
            return [],[], inference_time

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info("step 2:提取与question有关实体的文本的相关文本片段\n")
        text_entity_data_list = []
        userful_text_entity_data_list = []
        if len(related_entities)==1:
            entity_name, data_id = related_entities[0]
            text_entity_data = self.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
            userful_text_entity_data_list.append(text_entity_data)
            # text_entity_datas = self.kg_config.KG.get_entity_by_entityAndType(self, entity_name, 'text', self.kg_config.data_id_list)
            
            if text_entity_data and text_entity_data.type == 'text':
                related_snippet = await self.ExtractText(userful_information, question, text_entity_data.title, text_entity_data.text)
                text_entity_data_list.append((data_id, entity_name, related_snippet))
        else:
            text_tasks = []
            for entity_name, data_id in related_entities:
                text_entity_data = self.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
                userful_text_entity_data_list.append(text_entity_data)
                if not text_entity_data or text_entity_data.type != 'text':
                    continue
                
                # 异步任务
                text_tasks.append(self.ExtractText(userful_information, question, text_entity_data.title, text_entity_data.text))

            # 并发运行所有 ExtractText 任务
            start_time = time.time()
            text_results = await asyncio.gather(*text_tasks, return_exceptions=True)
            # self.base_config.logger.log_info(f"LLM 执行提取文本 time：{time.time()-start_time}")

            for text_entity_data, related_snippet in zip(related_entities, text_results):
                if isinstance(related_snippet, Exception):
                    # 如果该批次出现异常，可自行处理或跳过
                    self.base_config.logger.log_info(f"LLM 执行提取实体发生异常：{related_snippet}")
                    continue
                # print(f'text:{related_snippet}')
                if related_snippet and 'no related text snippet' not in related_snippet.lower():
                    text_entity_data_list.append((text_entity_data[1], text_entity_data[0], related_snippet))
        inference_time = time.time()-all_start_time
        # print(f'all text time:{time.time()-all_start_time}')
        return text_entity_data_list, userful_text_entity_data_list, inference_time
            
    async def ExtractTextEntities(self, question, userful_information, text_entity_data_list):
        start_time = time.time()
        judge_text_prompt = self.prompt_generator.create_judge_prompt(question, userful_information, text_entity_data_list)
        self.base_config.text_llm.init_message()
        llm_result, flag = await self.base_config.text_llm.predict_async(self.base_config.args, judge_text_prompt)
        if not flag:
            raise ValueError('openai content error')
        partial_related = self.text_processor.extract_entity_and_data_id(llm_result)

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{judge_text_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:\n{llm_result}\n")
            self.base_config.logger.log_info(f"Partial related_entities: {partial_related}\n")


        return partial_related, time.time()-start_time

    async def ExtractText(self, userful_information, question, title, text):

        reasoning_prompt =  self.prompt_generator.create_reasoning_prompt(userful_information, question, title, text)
        self.base_config.text_llm.init_message()
        llm_task = asyncio.create_task(self.base_config.text_llm.predict_async(self.base_config.args, reasoning_prompt))
        llm_result, flag = await llm_task
        if not flag:
            raise ValueError('openai content error')
        related_snippet = extract_output(llm_result)

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{reasoning_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
            self.base_config.logger.log_info(f"Related_snippet:{related_snippet}\n")

        if not isinstance(related_snippet, str):
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"LLM 执行提取文本异常：{llm_result}\n")
            # print('---------------text')
            # print(related_snippet)
        return related_snippet.upper()

