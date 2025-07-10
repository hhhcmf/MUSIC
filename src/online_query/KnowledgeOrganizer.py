class KnowledgeOrganizer:
    def __init__(self, base_config):
        self.base_config = base_config

    def organize_information(self, question, query_entity_information, query_table_information, image_qa_information):
        """组织信息并调用 LLM 生成组织后的结果"""
        self.base_config.logger.log_info('--------------------- Organizing Information ---------------------')
  
        organize_information_prompt = OrganizeInformationPrompt()
        prompt = organize_information_prompt.create_prompt(question, query_entity_information, query_table_information, image_qa_information)
        
        self.base_config.logger.log_info(f'User prompt:\n{prompt}')
        information = self.base_config.LLM.predict(self.base_config.args, prompt)
        self.base_config.logger.log_info(f'LLM response:\n{information}')
        
        return information
