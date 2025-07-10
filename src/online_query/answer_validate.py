
class AnswerValidate:
    def __init__(self, base_config, kg_config, else_config):
        self.base_config = base_config
        self.kg_config = kg_config
        self.else_config = else_config  

    def judge(self, question, answer):
        """判断答案是否充分并处理文本和图像信息"""
        
        # 获取实体信息
        entity_information, text_flag, text_answer_flag, image_title_flag = get_answer_entity(
            answer, self.kg_config.entity_link_hash, self.kg_config.KG, self.kg_config.kg_entities_name
        )

        judge_answer_prompt = JudgeAnswerPrompt()
        
        # 文本判断
        if text_flag:
            text_answer_flag, answer = self._judge_text_answer(
                question, answer, entity_information, text_flag, text_answer_flag, image_title_flag, judge_answer_prompt
            )
            if text_answer_flag:
                return answer

        # 图像判断
        if (not text_flag or (text_flag and not text_answer_flag)) and image_title_flag:
            return self._judge_image_answer(question, answer, entity_information, judge_answer_prompt)
        
        return answer

    def _judge_text_answer(self, question, answer, entity_information, text_flag, text_answer_flag, image_title_flag, judge_answer_prompt):
        """处理文本答案的判断逻辑"""
        
        # 初始化消息并创建提示词
        self.base_config.LLM.init_message()
        prompt = judge_answer_prompt.create_prompt(question, answer, entity_information, text_flag, text_answer_flag, image_title_flag)
        
        # 调用模型预测
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        
        # 日志记录
        self.base_config.logger.log_info('----------- Judge Answer (Text) ----------')
        self.base_config.logger.log_info(f'User prompt:\n{prompt}')
        self.base_config.logger.log_info(f'LLM response:\n{result}')

        # 判断答案是否足够
        if 'not enough' not in result.lower():
            text_answer_flag = True
            answer = extract_judge_text_answer(result.upper())
        else:
            text_answer_flag = False
        
        return text_answer_flag, answer

    def _judge_image_answer(self, question, answer, entity_information, judge_answer_prompt):
        """处理图像答案的判断逻辑"""
        
        # 初始化消息并创建提示词
        self.base_config.LLM.init_message()
        prompt = judge_answer_prompt.create_prompt(question, answer, entity_information, True, False, True)
        
        # 调用模型预测
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        
        # 日志记录
        self.base_config.logger.log_info('----------- Judge Answer (Image) ----------')
        self.base_config.logger.log_info(f'User prompt:\n{prompt}')
        self.base_config.logger.log_info(f'LLM response:\n{result}')

        # 判断是否需要继续
        if 'no need' in result.lower():
            return answer

        # 处理格式错误并提取推理结果
        post_result, flag = extract_reasoning(result)
        while not isinstance(post_result, dict):
            result = self._handle_format_error()
            post_result, flag = extract_reasoning(result)

        # 处理 API 调用并返回图像 QA 的结果
        return self._process_image_qa(post_result)

    def _handle_format_error(self):
        """处理模型响应中的格式错误"""
        
        prompt = "Output format error, please follow the provided output format."
        prompt += '''
Output format:
{
    "API Access": {
        "API": "<Interface name>",
        "Input": <JSON-formatted input>
    }
}
        '''
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        self.base_config.logger.log_info('-------------------------')
        self.base_config.logger.log_info(f'User prompt (Format Error Correction):\n{prompt}')
        self.base_config.logger.log_info(f'LLM response:\n{result}')
        return result

    def _process_image_qa(self, post_result):
        """根据 LLM 的响应调用图像 QA API 并返回结果"""
        
        api_output = ''
        for key, value in post_result.items():
            api = value.get('API')
            input_data = value.get('Input')
            title = input_data.get('title').upper()
            img_question = input_data.get('question')
            api_output, flag = ImageQA(self.kg_config.KG, title, img_question, self.else_config.img_folder)
            
            # 日志记录
            self.base_config.logger.log_info('----------- Image QA ----------')
            self.base_config.logger.log_info(f'Image title: {title}')
            self.base_config.logger.log_info(f'Question: {img_question}')
            self.base_config.logger.log_info(f'API Output:\n{api_output}')
        
        return api_output

# # 假设我们有如下初始化参数
# judge_llm = ...  # 用于判断答案的 LLM 模型
# logger = ...  # 日志记录器
# args = ...  # 其他参数
# KG = ...  # 知识图谱对象
# kg_entities_name = ...  # 知识图谱中的实体名称
# entity_link_hash = ...  # 实体链接哈希
# img_folder = ...  # 图像文件夹路径

# # 实例化 AnswerJudge
# answer_judge = AnswerJudge(judge_llm=judge_llm, logger=logger, args=args, KG=KG, kg_entities_name=kg_entities_name, entity_link_hash=entity_link_hash, img_folder=img_folder)

# # 使用 judge 方法判断答案
# final_answer = answer_judge.judge(question, answer)
