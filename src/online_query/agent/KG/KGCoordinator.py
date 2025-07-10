from online_query.agent.base import baseAgent
from online_query.agent.KG.prompt import KGReasonPrompt
from online_query.agent.table.TableParser_async import TableParser  
from online_query.agent.image.ImageAnalyzer_async import ImageAnalyzer
from online_query.agent.text.TextExtractor_async import TextExtractor
from utils.proprecss_llm_answer import extract_intermediate_answers
import time
import asyncio

class KGCoordinator(baseAgent):
    def __init__(self, base_config, kg_reason_processor):   
        super().__init__(base_config)
        self.kg_reason_processor = kg_reason_processor
        self.prompt_generator = KGReasonPrompt()
        self.table_parser = TableParser(base_config, kg_reason_processor.kg_config)
        self.image_analyzer = ImageAnalyzer(base_config, kg_reason_processor.kg_config)
        self.text_extractor = TextExtractor(base_config, kg_reason_processor.kg_config)
        
        
        self.table_records_list = []
        self.text_entity_data_list = []
        self.image_qa_list = []
        self.retrieved_entity_data_list = []
        self.entities_information_dict_list = []
    
    async def inference(self, question, question_embedding, userful_information, text_entity_data_list = [], image_qa_list = [], table_records_list = [], KDCI_flag = 1):
        if not KDCI_flag:
            entities_information_dict_list, intermediate_answer_entity_list, inference_time = self.inference_kg(question, question_embedding, userful_information)
            return entities_information_dict_list, intermediate_answer_entity_list, inference_time
        intermediate_answer_entity_list, get_intermediate_entity_time = await self.get_intermediate_answer_entity_list(question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list)
        intermediate_answer_entity_data_list, data_inference_time = await self.get_intermediate_entity_data_information(question, userful_information, intermediate_answer_entity_list)
        inference_time = data_inference_time + get_intermediate_entity_time
        return intermediate_answer_entity_data_list, inference_time


    async def get_intermediate_entity_data_information(self, question, userful_information, intermediate_answer_entity_list):
        intermediate_answer_entity_data_list = []
        for entity_name in intermediate_answer_entity_list:
            intermediate_answer_entity_data_list += self.kg_reason_processor.get_entity_data_by_name(entity_name)
        self.retrieved_entity_data_list += intermediate_answer_entity_data_list

        self.base_config.logger.log_info('-------------在获取中间实体中进行获取模态数据-------')
        entity_data_list = self.retrieved_entity_data_list
        table_records_data_list, image_qa_list, text_entity_data_list, data_inference_time, data_inference_time_details = await self.get_agent_related_data(question, userful_information, intermediate_answer_entity_data_list)
        
        return intermediate_answer_entity_data_list, data_inference_time

    async def get_intermediate_answer_entity_list(self, question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list):
        """
        并行执行表格推理、图像推理和文本推理，提高推理效率。
        """
        # 结果存储
        intermediate_table_entity_list, intermediate_image_entity_list, intermediate_text_entity_list = [], [], []

        tasks = {}

        if self.base_config.args.engine == 'gpt-4o':
            gpt_4_flag = 1
        else:
            gpt_4_flag = 0

        if table_records_list and table_records_list[0][0]: 
            tasks['table'] = asyncio.create_task(
                self.reason_table(question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list, gpt_4_flag))
            # tasks.append(asyncio.to_thread(self.reason_table, question, userful_information, text_entity_data_list, image_qa_list, table_records_list))

        if image_qa_list:
            tasks['image'] = asyncio.create_task(self.reason_image( question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list, gpt_4_flag))
            # tasks.append(asyncio.to_thread(self.reason_image, question, userful_information, text_entity_data_list, image_qa_list, table_records_list))

        if text_entity_data_list:
            tasks['text'] = asyncio.create_task(self.reason_text(question, question_embedding,userful_information, text_entity_data_list, image_qa_list, table_records_list, gpt_4_flag))
            # tasks.append(asyncio.to_thread(self.reason_text, question, userful_information, text_entity_data_list, image_qa_list, table_records_list)

        # **并行执行所有任务**
        start_time = time.time()

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        get_intermediate_entity_time = time.time() - start_time

        # 确保结果按照任务的 key 进行存储
        if table_records_list and table_records_list[0][0]:
            intermediate_table_entity_list = tasks.get("table") and results[list(tasks.keys()).index("table")]
        if image_qa_list:
            intermediate_table_entity_list = tasks.get("image") and results[list(tasks.keys()).index("image")]
        if text_entity_data_list:
            intermediate_text_entity_list = tasks.get("text") and results[list(tasks.keys()).index("text")]
       
       
        intermediate_table_entity_list = intermediate_table_entity_list if isinstance(intermediate_table_entity_list, list) else []
        intermediate_image_entity_list = intermediate_image_entity_list if isinstance(intermediate_image_entity_list, list) else []
        intermediate_text_entity_list = intermediate_text_entity_list if isinstance(intermediate_text_entity_list, list) else []

        intermediate_answer_entity_list = list(set(
            intermediate_table_entity_list + intermediate_image_entity_list + intermediate_text_entity_list
        ))
        self.base_config.logger.log_info(f'得到的所有中间实体:{intermediate_answer_entity_list}')

        return intermediate_answer_entity_list, get_intermediate_entity_time

   

    async def reason_table(self, question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_data_list, gpt_4_flag):
        '''
        基于table数据找到中间实体
        userful_information:

        '''
        self.base_config.logger.log_info("---基于表格数据找中间答案实体---\n")
        result_list = []
        inference_time = 0
        # self.base_config.logger.log_info(f"reason_table:table_records:{table_records_data_list}\n")
        for table_records_data in table_records_data_list:
            table_records_list = [table_records_data]
            agent_response_information, passage_information, image_qa_information, table_record_information = self.kg_reason_processor.get_agent_response_information(question, question_embedding, text_entity_data_list, image_qa_list, table_records_list)
            
            table_reason_prompt = self.prompt_generator.create_table_reason_prompt(question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag)
            self.base_config.table_llm.init_message()

            result, flag = self.base_config.table_llm.predict(self.base_config.args, table_reason_prompt)
            if not flag:
                raise ValueError('openai content error')
            intermediate_answer_entity_list = extract_intermediate_answers(result)
            if 'no intermediate answer' in result.lower():
                intermediate_answer_entity_list = []
            for entity_name in intermediate_answer_entity_list:
                if entity_name not in table_record_information:
                    intermediate_answer_entity_list.remove(entity_name)
            result_list += intermediate_answer_entity_list
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"User prompt:{table_reason_prompt}\n")
                self.base_config.logger.log_info(f"LLM response:{result}\n")
                self.base_config.logger.log_info(f"Intermediate_answers:{intermediate_answer_entity_list}\n")
        
        return list(set(result_list))

    async def reason_text(self, question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list, gpt_4_flag):
        '''
        基于text数据找到中间实体
        userful_information:

        '''
        self.base_config.logger.log_info("---基于文本数据找中间答案实体---\n")
    
        agent_response_information, passage_information, image_qa_information, table_record_information = self.kg_reason_processor.get_agent_response_information(question, question_embedding, text_entity_data_list, image_qa_list, table_records_list)
        
        text_reason_prompt = self.prompt_generator.create_text_reason_prompt(question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag)
        self.base_config.text_llm.init_message()
        result, flag = self.base_config.text_llm.predict(self.base_config.args, text_reason_prompt)
        if not flag:
            raise ValueError('openai content error')
        intermediate_answer_entity_list = extract_intermediate_answers(result)
        if 'no intermediate answer' in result.lower():
            intermediate_answer_entity_list = []
        for entity_name in intermediate_answer_entity_list:
            if entity_name not in passage_information:
                intermediate_answer_entity_list.remove(entity_name)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{text_reason_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{result}\n")
            self.base_config.logger.log_info(f"Intermediate_answers:{intermediate_answer_entity_list}\n")
        
        return intermediate_answer_entity_list

    async def reason_image(self, question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_list, gpt_4_flag):
        '''
        基于image数据找到中间实体
        userful_information:

        '''
        self.base_config.logger.log_info("---基于图像数据找中间答案实体---\n")
    
        agent_response_information, passage_information, image_qa_information, table_record_information = self.kg_reason_processor.get_agent_response_information(question, question_embedding, text_entity_data_list, image_qa_list, table_records_list)
        
        
        image_reason_prompt = self.prompt_generator.create_image_reason_prompt(question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag)
        self.base_config.image_llm.init_message()
        result, flag = self.base_config.image_llm.predict(self.base_config.args, image_reason_prompt)
        if not flag:
            raise ValueError('openai content error')
        intermediate_answer_entity_list = extract_intermediate_answers(result)
        if 'no intermediate answer' in result.lower():
            intermediate_answer_entity_list = []
        for entity_name in intermediate_answer_entity_list:
            if entity_name not in image_qa_information:
                intermediate_answer_entity_list.remove(entity_name)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{image_reason_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{result}\n")
            self.base_config.logger.log_info(f"Intermediate_answers:{intermediate_answer_entity_list}\n")
        
        return intermediate_answer_entity_list



    async def get_agent_related_data(self, question, userful_information, entity_data_list, summarize_flag = 0):
        image_entity_data_list, table_entity_data_list, text_entity_data_list = [], [], []


        # Step 1: 解析数据实体
        start_time = time.time()
        for entity_data in entity_data_list:
            modality_type = entity_data.type
            if modality_type == 'image':
                title = entity_data.title

                
                image_entity_data = entity_data
                if image_entity_data and (image_entity_data.title == image_entity_data.entity_name or image_entity_data.isImageTitle):
                    # self.retrieved_entity_data_list.append(image_entity_data)
                    # self.retrieved_entity_data_list += self.kg_reason_processor.kg_config.KG.get_entity_by_entity(
                    #     image_entity_data.title, self.kg_reason_processor.kg_config.data_id_list
                    # )
                    image_entity_data_list.append(image_entity_data)
                else:
                    image_entity_data = self.kg_reason_processor.kg_config.KG.get_image_entity_data(
                    title, self.kg_reason_processor.kg_config.data_id_list
                    )
                    if image_entity_data and (image_entity_data.title == image_entity_data.entity_name or image_entity_data.isImageTitle):
                        image_entity_data_list.append(image_entity_data)

            elif modality_type == 'table':
                table_entity_data_list.append(entity_data)
                # self.retrieved_entity_data_list.append(entity_data)

            else:
                text_entity_data_list.append(entity_data)
                # self.retrieved_entity_data_list.append(entity_data)

        table_entity_data_list = list(set(table_entity_data_list))
        image_entity_data_list = list(set(image_entity_data_list))
        text_entity_data_list = list(set(text_entity_data_list))
        data_pretprocess_time = time.time() - start_time

        result = await self.image_analyzer.inference(question, userful_information, image_entity_data_list, summarize_flag)
        print('test:', result)
        
        # Step 2: 定义并行任务
        start_time = time.time()
        tasks = {}
        if table_entity_data_list:
            tasks["table"] = asyncio.create_task(
                self.table_parser.inference(question, userful_information, table_entity_data_list)
            )
        if image_entity_data_list:
            tasks["image"] = asyncio.create_task(
                self.image_analyzer.inference(question, userful_information, image_entity_data_list, summarize_flag)
            )
        if text_entity_data_list:
            tasks["text"] = asyncio.create_task(
                self.text_extractor.inference(question, userful_information, text_entity_data_list)
            )

        start_time = time.time()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        total_inference_time = time.time() - start_time

        table_results = tasks.get("table") and results[list(tasks.keys()).index("table")]
        image_results = tasks.get("image") and results[list(tasks.keys()).index("image")]
        text_results = tasks.get("text") and results[list(tasks.keys()).index("text")]

    
        # 处理表格数据
        # print('----------')
        # print(type(table_results))
        # print(table_results, type(table_results))
        table_records_data_list, userful_table_entity_data_list, table_inference_time = table_results or ([], [], 0)
        if table_records_data_list:
            # print(type(table_records_data_list)) 
            # print(table_records_data_list)
            self.table_records_list.extend(table_records_data_list)
            

        # 处理图像数据
        print('----------')
        print(type(image_results))
        print('image', image_results, type(image_results))
        image_question_list, userful_image_entity_data_list, judge_object_question, image_inference_time = image_results or ([], [], '', 0)
        if image_question_list:
            # print(image_question_list)
            self.image_qa_list.extend(image_question_list)
            self.image_qa_list = list(set(self.image_qa_list))

        # 处理文本数据
        # print('----------')
        # print(type(text_results))
        # print(text_results, type(text_results))
        text_data_list, userful_text_entity_data_list, text_inference_time = text_results or ([], [], 0)
        if text_data_list:
            # print('-------', text_data_list)
            self.text_entity_data_list.extend(text_data_list)
            self.text_entity_data_list = list(set(self.text_entity_data_list))

        # 统一去重
        data_postprocess_time = time.time() - start_time
        self.retrieved_entity_data_list = list(set(userful_image_entity_data_list + userful_table_entity_data_list + userful_text_entity_data_list))
        # print(f'data pretprocess time: {data_pretprocess_time:.2f} seconds')
        # print(f'create task time: {create_task_time:.2f} seconds')
        # print(f'Total inference time: {total_inference_time:.2f} seconds')
        # print(f'data postprocess time: {data_postprocess_time:.2f} seconds')

        inference_time = max(max(table_inference_time,image_inference_time),text_inference_time)
        inference_time_details = {
            'table':table_inference_time,
            'text':text_inference_time,
            'image':image_inference_time
        }

        # self.base_config.logger.log_info(f'各模态推理结果:\ntable:\n{table_records_data_list}\ntext:\n{text_data_list}\nimage:\n{image_question_list}')
        # self.base_config.logger.log_info(f'data_inference_time_details:table:{table_inference_time}\ntext:{text_inference_time}\nimage:{image_inference_time}\n')

        return  list(set(table_records_data_list)), list(set(image_question_list)), list(set(text_data_list)), inference_time, inference_time_details


    def inference_kg(self, question, question_embedding, useful_information):
        entities_information_dict_list, inference_time = {}, 0
        intermediate_answer_entity_list, inference_time = self.reason_kg(question, useful_information)
        entities_information_dict_list, retrieved_entity_data_list =  self.kg_reason_processor.get_query_entities_information_dict(question, intermediate_answer_entity_list, question_embedding)
        for entities_information_dict in entities_information_dict_list:
            if entities_information_dict in self.entities_information_dict_list :
                continue
            else:
                self.entities_information_dict_list.append(entities_information_dict)
        self.retrieved_entity_data_list += retrieved_entity_data_list
        self.retrieved_entity_data_list = list(set(self.retrieved_entity_data_list))
        return self.entities_information_dict_list, intermediate_answer_entity_list, inference_time

    def reason_kg(self, question, useful_information):
        start_time = time.time()
        kg_reason_prompt = self.prompt_generator.create_kg_reason_prompt(question, useful_information)
        self.base_config.LLM.init_message()
        result, flag = self.base_config.LLM.predict(self.base_config.args, kg_reason_prompt)
        if not flag:
            raise ValueError('openai content error')
        intermediate_answer_entity_list = extract_intermediate_answers(result)
        if 'no intermediate answer' in result.lower():
            intermediate_answer_entity_list = []
        reason_kg_time = time.time() - start_time
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{kg_reason_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{result}\n")
            self.base_config.logger.log_info(f"Intermediate_answers:{intermediate_answer_entity_list}\n")
        return intermediate_answer_entity_list, reason_kg_time
    

    async def get_related_information_from_entity_data_list(self, question, entity_data_list, question_embedding):
        kg_entities_information = ''
        if not entity_data_list:
            return ''
        if len(entity_data_list) < 6:
            kg_entities_information = self.kg_reason_processor.get_kg_entities_information(question, entity_data_list, question_embedding)
            return kg_entities_information

        def chunkify(lst, chunk_size=5):
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        chunks = list(chunkify(entity_data_list, 5))
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(self.get_related_entity_from_chunk(question, chunk, question_embedding))
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        related_entity_data_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if self.base_config.args.SHOW_LOGGER:
                    self.base_config.logger.log_info(f"提取相关实体发生异常：{result}")
                continue
            for entity_name, data_id in result:
                entity_data = self.kg_reason_processor.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
                related_entity_data_list.append(entity_data)
        
        if related_entity_data_list:
            kg_entities_information = self.kg_reason_processor.get_kg_entities_information(question, related_entity_data_list, question_embedding)
        return kg_entities_information



        
    async def get_related_entity_from_chunk(self, question, entity_data_list, question_embedding):
        entity_information = self.kg_reason_processor.get_kg_entities_information(question, entity_data_list, question_embedding)
        get_related_prompt = self.prompt_generator.create_related_entity_prompt(question, entity_information)
        llm_result, flag = await self.base_config.LLM.predict_async(self.base_config.args, get_related_prompt)
        if not flag:
            raise ValueError('openai content error')
        if 'no relevant entity' in llm_result.lower():
            return {}
        related_entities = self.text_extractor.text_processor.extract_entity_and_data_id(llm_result)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{get_related_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
            self.base_config.logger.log_info(f"Related_entities:{related_entities}\n")

        return related_entities

    def get_releted_entity_information(self, userful_information, question, question_embedding, entity_name, data_id_list, KDCI_flag = 1):
        '''
        获取与指定实体的数据, 包括实体信息、文本、图像以及表格
        '''
        # if entity_name not in self.kg_config.kg_entities_name:

        entity_information = '' 
        retrieved_entity_data_list = []
        entity_information_list = []
        entity_name_list = [entity_name.upper()]
        if entity_name not in self.kg_reason_processor.kg_config.kg_entities_name:
            entity_name_list = entity_name.split(',')
            if len(entity_name_list) == 1:
                entity_information = ''
                table_records_data, image_qa_list, text_entity_data_list = None, [], []

        else:
            for entity_name in entity_name_list:
                entity_data_list = self.kg_reason_processor.kg_config.KG.get_entity_by_entity(entity_name.upper(), data_id_list)
                for idx, entity_data in enumerate(entity_data_list):
                    entity_information_list.append(f'{idx+1}.description:{entity_data.description}')
                entity_information += f'**{entity_name} information:**\n'+'\n'.join(entity_information_list)
            
            # table_records_data, image_qa_list, text_entity_data_list = self.get_agent_related_data(question, userful_information, entity_data_list)
        if not KDCI_flag:
            return entity_information
        agent_response_information, _, _, _ = self.kg_reason_processor.get_agent_response_information(question, question_embedding, self.text_entity_data_list, self.image_qa_list, self.table_records_list)

        return entity_information, agent_response_information, self.table_records_list, self.image_qa_list, self.text_entity_data_list
        
    async def get_related_entity_informtions(self, question, entity_information_dict_list):
        self.base_config.logger.log_info('查找相关实体信息')
        entity_information = ''
        # if len(entity_information_dict_list)<3:
        #     for entity_information_dict in entity_information_dict_list:
        #         for query_entity, information in entity_information_dict.items():
        #             # entity_information += f'Related to {query_entity}:\n{information["entity_information"]}\n'
        #             entity_information += f'{information["entity_information"]}\n'
        #     return entity_information

        def chunkify(lst, chunk_size=5):
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        chunks = list(chunkify(entity_information_dict_list, 5))
        tasks = []
        # await self.get_related_entity(question, chunks[0])
        for i, chunk in enumerate(chunks):
            tasks.append(self.get_related_entity(question, chunk))

        related_entity_information_dict_list = []
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if self.base_config.args.SHOW_LOGGER:
                    self.base_config.logger.log_info(f"提取相关实体发生异常：{result}")
                continue
            
            related_entity_information_dict_list.extend(result) 
        pos = 1
        record = []
        for related_entity_information_dict in related_entity_information_dict_list:
            for entity_entity_data_id, value in related_entity_information_dict.items():
                entity_name, data_id = entity_entity_data_id
                if (entity_name, data_id) in record:
                    continue
                else:
                    record.append((entity_name, data_id))
                entity_information += f"{pos}.{value['entity_information']}\n"
        return entity_information

    
    async def get_related_entity(self, question, entity_information_dict_list):
        entities_information = ''
        all_dict = {}
        pos = 1
        for entity_information_dict in entity_information_dict_list:
            for query_entity, value in entity_information_dict.items():
                information = value.get('entity_information')
                # for retrieve_entity, retrieve_entity_information in value.items():
                #     if retrieve_entity != 'entity_information' and retrieve_entity != 'entity_data_list':
                        # print(retrieve_entity)
                all_dict[query_entity] = {}
                all_dict[query_entity]['entity_information']= information
                all_dict[query_entity]['belong'] = entity_information_dict
                entities_information += f'{pos}.{information}\n'
                pos += 1
        get_related_prompt = self.prompt_generator.create_related_entity_prompt(question, entities_information)
        llm_result, flag = await self.base_config.LLM.predict_async(self.base_config.args, get_related_prompt)
        related_entities = []
        if 'no relevant entity' in llm_result.lower():
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"User prompt:{get_related_prompt}\nLLM response:{llm_result}\nRelated_entities:{related_entities}\n")
            return ''
        if not flag:
            raise ValueError('openai content error')
        related_entities = self.text_extractor.text_processor.extract_entity_and_data_id(llm_result)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"User prompt:{get_related_prompt}\nLLM response:{llm_result}\nRelated_entities:{related_entities}\n")
          
        related_entities_information = ''
        related_entities_information_dict = {}
        related_entities_information_dict_list = []
        for entity_name, data_id in related_entities:
            entity_data = self.kg_reason_processor.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
            if entity_data not in self.retrieved_entity_data_list:
                self.retrieved_entity_data_list.append(entity_data)
            # related_entities_information += all_dict[entity_name]['entity_information']
            if entity_name in all_dict.keys():
                related_entities_information_dict[(entity_data.entity_name, entity_data.data_id)] = all_dict[entity_name]
                related_entities_information_dict_list.append(related_entities_information_dict)
                if all_dict[entity_name]['belong'] not in self.entities_information_dict_list:
                    self.entities_information_dict_list.append(all_dict[entity_name]['belong'])
        # self.base_config.logger.log_info(f"related_entities_information:{related_entities_information}\n")
        return related_entities_information_dict_list