class ReasoningEngine:
    def __init__(self, llm, further_llm, logger, retrieve_llm, args, kg, kg_entities_name, embed_model, embed_tokenizer, db_path, img_folder, entity_link_hash):
        """
        :param llm: 用于初始推理的 LLM 模型
        :param further_llm: 用于进一步推理的 LLM 模型
        :param logger: 日志记录器
        :param retrieve_llm: 实体查询 LLM
        :param args: 其他参数
        :param kg: 知识图谱
        :param kg_entities_name: 知识图谱中的实体名称
        :param embed_model: 嵌入模型
        :param embed_tokenizer: 嵌入模型的 Tokenizer
        :param db_path: 数据库路径
        :param img_folder: 图像文件夹路径
        :param entity_link_hash: 实体链接哈希
        """
        self.llm = llm
        self.further_llm = further_llm
        self.logger = logger
        self.retrieve_llm = retrieve_llm
        self.args = args
        self.kg = kg
        self.kg_entities_name = kg_entities_name
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.db_path = db_path
        self.img_folder = img_folder
        self.entity_link_hash = entity_link_hash

    def start_reasoning(self, question, entity_info, table_info, entity_apis, table_apis, image_apis):
        """执行初始推理"""
        self.logger.log_info('****************** Start Reasoning *******************')
        self.llm.init_message()
        prompt = ReasoningPrompt().create_prompt(question, entity_info, table_info, entity_apis, table_apis, image_apis)
        output = self.llm.predict(self.args, prompt)
        self.logger.log_info(f'User prompt:\n{prompt}')
        self.logger.log_info(f'LLM response:\n{output}')
        return output

    def further_reasoning(self, question, table_reasoning_prompt, image_reasoning_prompt, entity_reasoning_prompt, question_embedding):
        """执行进一步推理"""
        further_prompt = FurtherReasoningPrompt().create_prompt(question, table_reasoning_prompt, image_reasoning_prompt, entity_reasoning_prompt)
        self.further_llm.init_message()
        result = self.further_llm.predict(self.args, further_prompt)
        self.logger.log_info(f'Further reasoning prompt:\n{further_prompt}')
        self.logger.log_info(f'Further reasoning LLM response:\n{result}')
        
        if 'no further interface calls' not in result.lower():
            return self.process_reasoning_result(self.further_llm, question, question_embedding, result)
        return result

    def process_reasoning_result(self, llm, question, question_embedding, result):
        """处理推理结果并根据 LLM 输出调用相关 API"""
        post_result, flag = extract_reasoning(result)
        
        while not isinstance(post_result, dict):
            result = self.handle_format_error(llm)
            post_result, flag = extract_reasoning(result)

        table_reasoning, image_reasoning, entity_reasoning = '', '', ''
        api_output, information, table_info, entity_info, image_info = '', '', '', '', ''
        
        # 遍历推理结果并处理各类 API 请求
        for key, value in post_result.items():
            api = value.get('API')
            input_data = value.get('Input')
            if api == 'QueryEntity':
                api_output = self._handle_query_entity(input_data, question, question_embedding)
            elif api == 'QueryTable':
                api_output = self._handle_query_table(input_data)
            elif api == 'ImageQA':
                api_output = self._handle_image_qa(input_data)

            # 日志记录和信息整理
            information += f'API Output for {api}:\n{api_output}\n'
        
        return table_reasoning, image_reasoning, entity_reasoning, information

    def handle_format_error(self, llm):
        """处理输出格式错误的响应"""
        prompt = "Output format error, please follow the provided output format.\n" + '''
        Output format:
        {
            "API Access": {
                "API": "<Interface name>",
                "Input": <JSON-formatted input>
            }
        }
        '''
        result = llm.predict(self.args, prompt)
        self.logger.log_info(f'User prompt (Format Error Correction):\n{prompt}')
        self.logger.log_info(f'LLM response:\n{result}')
        return result

    def _handle_query_entity(self, input_data, question, question_embedding):
        """处理 QueryEntity API 请求"""
        entity_name = input_data.get('entity_name').upper()
        self.retrieve_llm.init_message()
        table_data, image_data, text_data, api_output, flag = QueryEntity(
            entity_name, question, question_embedding, self.entity_link_hash, 
            self.retrieve_llm, self.kg, self.kg_entities_name, self.args, 
            self.logger, self.logger, self.args.embedding_model_name, 
            self.embed_model, self.embed_tokenizer, self.entity_index, 
            self.entity_id_to_key, self.entity_desp_index, 
            self.entity_desp_id_to_key, self.triple_index, 
            self.triple_id_to_key, self.triple_desp_index, 
            self.triple_desp_id_to_key, self.top_entity_k, 
            self.top_triple_k
        )
        return api_output

    def _handle_query_table(self, input_data):
        """处理 QueryTable API 请求"""
        table_name = input_data.get('table name').upper()
        select_column = input_data.get('select column')
        conditions = input_data.get('conditions', {})
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            api_output, flag = QueryTable(table_name, select_column, conditions, cursor)
        return api_output

    def _handle_image_qa(self, input_data):
        """处理 ImageQA API 请求"""
        title = input_data.get('title').upper()
        img_question = input_data.get('question')
        api_output, flag = ImageQA(self.kg, title, img_question, self.img_folder)
        return api_output

# llm = ...  # 用于初始推理的 LLM 模型
# further_llm = ...  # 用于进一步推理的 LLM 模型
# logger = ...  # 日志记录器
# retrieve_llm = ...  # 实体查询 LLM
# args = ...  # 其他参数
# kg = ...  # 知识图谱对象
# kg_entities_name = ...  # 知识图谱中的实体名称
# embed_model = ...  # 嵌入模型
# embed_tokenizer = ...  # 嵌入模型的 Tokenizer
# db_path = "path_to_db"
# img_folder = "path_to_images"
# entity_link_hash = ...

# # 实例化 ReasoningEngine
# reasoning_engine = ReasoningEngine(
#     llm=llm, further_llm=further_llm, logger=logger, retrieve_llm=retrieve_llm,
#     args=args, kg=kg, kg_entities_name=kg_entities_name, embed_model=embed_model,
#     embed_tokenizer=embed_tokenizer, db_path=db_path, img_folder=img_folder,
#     entity_link_hash=entity_link_hash
# )

# # 调用推理方法
# initial_result = reasoning_engine.start_reasoning(
#     question, entity_information, table_information, query_entity_apis, query_table_apis, image_qa_apis
# )

# # 进一步推理
# further_result = reasoning_engine.further_reasoning(
#     question, tableReasoningPrompt, imageTitleReasoningPrompt, entityReasoningPrompt, question_embedding
# )
