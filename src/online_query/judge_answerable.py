class AnswerabilityJudge:
    def __init__(self, base_config):
        self.base_config = base_config
        

    def is_answerable(self, question):
        """判断问题是否可回答"""
        self.base_config.logger.log_info('--------------------- Judge Answerability ---------------------')

        judge_answerable_prompt = AnswerablePrompt()
        prompt = judge_answerable_prompt.create_prompt(question)
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        
        self.base_config.logger.log_info(f'User prompt:\n{prompt}')
        self.base_config.logger.log_info(f'LLM response:\n{result}')

        answerable = 'yes' in result.lower()
        return answerable
