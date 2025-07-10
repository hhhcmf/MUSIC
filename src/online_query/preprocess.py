from utils.proprecss_llm_answer import extract_time_entities, extract_entities
from prompt.prompt_generator import ExtractTimePrompt, ExtractQuestionEntityPrompt
class PreprocessPhase:
    def __init__(self, base_config):
        self.base_config = base_config
    
    def run(self, question):
        max_tries = 5 
        entities = []

        # Step 1: 提取时间实体
        self.base_config.logger.log_info('--------------------- Step 1.1: Extract question entity ---------------')
        time_entities = self.extract_time_entities(question)
        entities.extend(time_entities)

        # Step 2: 提取其他实体
        self.base_config.logger.log_info('--------------------- Step 1.2: Extract additional entities ---------------')
        entities += self.extract_question_entities(question, max_tries)

        unique_entities = list(set(entities))
        self.base_config.logger.log_info(f'Final question entities: {unique_entities}')
        return unique_entities

    def extract_time_entities(self, question):
        """Extract time-related entities from the question."""
        extract_time_prompt = ExtractTimePrompt()
        prompt = extract_time_prompt.create_prompt(question)
        try:
            self.base_config.LLM.init_message
            result = self.base_config.LLM.predict(self.base_config.args, prompt)
            time_entities = extract_time_entities(result.upper())
            self.base_config.logger.log_info(f'User prompt: {prompt}')
            self.base_config.logger.log_info(f'LLM response (time): {time_entities}')
        except Exception as e:
            self.base_config.logger.log_error(f'Error during time entity extraction: {e}')
            time_entities = []
        return time_entities

    def extract_question_entities(self, question, max_tries):
        """Extract general entities from the question with retries."""
        entities = []
        extract_entity_prompt = ExtractQuestionEntityPrompt()
        for attempt in range(max_tries):
            try:
                self.base_config.logger.log_info(f'Attempt {attempt + 1}/{max_tries} for entity extraction')
                self.base_config.LLM.init_message()  # Reset conversation
                prompt = extract_entity_prompt.create_prompt(question)
                result = self.base_config.LLM.predict(self.base_config.args, prompt)
                extracted_entities = extract_entities(result.upper())
                entities.extend(extracted_entities)
                self.base_config.logger.log_info(f'User prompt: {prompt}')
                self.base_config.logger.log_info(f'LLM response (entities): {extracted_entities}')
            except Exception as e:
                self.base_config.logger.log_error(f'Error during entity extraction attempt {attempt + 1}: {e}')
        return entities


