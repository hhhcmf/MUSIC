class QuestionAnswerer:
    def __init__(self, base_config):
        self.base_config = base_config

    def get_answer(self, question, entity_information):
        """生成问题的答案"""
        self.base_config.logger.log_info('--------------------- Question Answering ---------------------')

        answer_prompt = AnswerPrompt()
        prompt = answer_prompt.create_prompt(question, entity_information)
        
        result = self.base_config.LLM.predict(self.base_config.args, prompt)

        self.base_config.logger.log_info(f'User prompt:\n{prompt}')
        self.base_config.logger.log_info(f'LLM response:\n{result}')
        answer = extract_answer(result)
        return answer
