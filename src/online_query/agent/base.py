from utils.proprecss_llm_answer import extract_reasoning, fix_conditions
class baseAgent:
    def __init__(self, base_config):
        self.base_config = base_config
    
    def _deal_llm_result(self, result):
        try:
            post_result, flag = extract_reasoning(result)
        except Exception:
            flag = False
        while not isinstance(post_result, dict) or not flag:
            try:
                post_result = fix_conditions(result)
                flag = True
            except Exception:
                flag = False

            if not isinstance(post_result, dict) or not flag:
                prompt = "Output format error, please follow the provided output format(no any explanation, no ```json```):"
                prompt += '''
{
    "API Access": {
        "API": "<Interface name>",
        "Input": <JSON-formatted input>
    }
}
                '''
                result = self.base_config.LLM.predict(self.base_config.args, prompt)
                self.base_config.logger.log_info('-------------------------')
                self.base_config.logger.log_info(f'user prompt:\n{prompt}')
                self.base_config.logger.log_info(f'LLM response:\n{result}')
                post_result, flag = extract_reasoning(result)
        
        return post_result

class BasePromptGenerator:
    """
    Base class for generating prompts.

    This class provides a template method pattern for generating prompts. Subclasses
    should implement the `_create_prompt` method to provide specific prompt generation logic.

    Methods
    -------
    create_prompt(*args, **kwargs)
        Generates a prompt by calling the `_create_prompt` method.
    """
    def __init__(self, *args, **kwargs):
        pass

    def create_reasoning_prompt(self, *args, **kwargs):
        return self._create_reasoning_prompt(*args, **kwargs)

    def _create_reasoning_prompt(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement create_reasoning_prompt.")
    
    def create_judge_prompt(self, *args, **kwargs):
        return self._create_judge_prompt(*args, **kwargs)

    def _create_judge_prompt(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement create_judge_prompt.")
