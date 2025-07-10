from utils.proprecss_llm_answer import extract_time_entities, extract_entities, extract_answer, extract_output_reason, extract_answer_text, extract_output
from prompt.prompt_generator import ExtractTimePrompt, ExtractQuestionEntityPrompt, OrganizeInformationPrompt, AnswerablePrompt, QuestionAnswerPrompt, ValidateSemanticsPrompt, ValidateAnswerInformationPrompt, QuestionAnswerByDataPrompt, QuestionAnswerPipelinePrompt, QuestionAnswerLittlePrompt, SimplyAnswerPrompt, AnswerRewritePrompt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
class PreprocessPhase:
    def __init__(self, base_config):
        self.base_config = base_config
    
    def run(self, orginal_question, question):
        max_tries = 3 
        entities = []
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'question:{question}')
            self.base_config.logger.log_info('---------------------Step 1 提取问题实体---------------------\n')
            self.base_config.logger.log_info('---1.1: Extract question entity ---\n')
        time_entities = self.extract_time_entities(question)
        entities.extend(time_entities)

        # Step 2: 提取其他实体
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info('---1.2: Extract additional entities ---\n')
        entities += self.extract_question_entities(question, max_tries)
        # entities += self.extract_question_entities(orginal_question, max_tries)
        
        unique_entities = list(set(entities))
        self.base_config.logger.log_info(f'**Final question entities:** {unique_entities}\n')
        return unique_entities

    def extract_time_entities(self, question):
        """Extract time-related entities from the question."""
        extract_time_prompt = ExtractTimePrompt()
        prompt = extract_time_prompt.create_prompt(question)
        try:
            self.base_config.LLM.init_message()
            result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
            if not flag:
                raise ValueError('openai content error')
            time_entities = extract_time_entities(result.upper())
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f'User prompt: {prompt}')
                self.base_config.logger.log_info(f'LLM response (time): {time_entities}')
        except Exception as e:
            self.base_config.logger.log_error(f'Error during time entity extraction: {e}\n')
            time_entities = []
        return time_entities

    def extract_question_entities(self, question, max_tries):
        """Extract general entities from the question with retries."""
        entities = []
        extract_entity_prompt = ExtractQuestionEntityPrompt()
        for attempt in range(max_tries):
            try:
                # self.base_config.logger.log_info(f'Attempt {attempt + 1}/{max_tries} for entity extraction')
                self.base_config.LLM.init_message()  # Reset conversation
                prompt = extract_entity_prompt.create_prompt(question)
                result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
                
                extracted_entities = extract_entities(result.upper())
                entities.extend(extracted_entities)
                # self.base_config.logger.log_info(f'User prompt: {prompt}')
                # self.base_config.logger.log_info(f'LLM response (entities): {extracted_entities}')
            except Exception as e:
                if not entities:
                    attempt += 1
                self.base_config.logger.log_error(f'Error during entity extraction attempt {attempt + 1}: {e}\n')
        if not entities:
            raise ValueError('openai content error')
        return entities

class KnowledgeOrganizerPhase:
    def __init__(self, base_config):
        self.base_config = base_config
    
    # def run(self, question, text_data_list, table_records_list, image_qa_list):
    #     userful_information = ''
    #     idx = 1
    #     for text_data in text_data_list:
            

    def run(self, question, query_entity_information, agent_response_information, KDCI_flag = 1):
        """组织信息并调用 LLM 生成组织后的结果"""
        self.base_config.logger.log_info('--------------------- Organizing Information ---------------------\n')
  
        organize_information_prompt = OrganizeInformationPrompt()
        prompt = organize_information_prompt.create_prompt(question, query_entity_information, agent_response_information, KDCI_flag)
        self.base_config.LLM.init_message()
        information, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
        self.base_config.logger.log_info(f'LLM response:\n{information}\n')
        
        return information

class AnswerRewritePhase:
    def __init__(self, base_config):
        self.base_config = base_config

    def run(self, question, answer):
        self.base_config.logger.log_info('--------------------- 答案重写 ---------------------\n')
        answer_rewrite_prompt = AnswerRewritePrompt()
        prompt = answer_rewrite_prompt.create_prompt(question, answer)
        try_count = 0
        flag = False
        while try_count < 3 and not flag:
            try_count += 1
            self.base_config.LLM.init_message()
            result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        # if not flag:
        #     raise ValueError('openai content error')
            if self.base_config.args.SHOW_LOGGER: 
                self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
                self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        answer = extract_answer(result)
        return answer

        # answerable = self.judge_answerable(question)
class AnswerGeneratePhase:
    def __init__(self, base_config, kg_config):
        self.base_config = base_config
        self.kg_config = kg_config

    def run(self, question, useful_information, short_flag):
        
        self.base_config.logger.log_info('--------------------- 答案生成 ---------------------\n')
        # answerable = self.judge_answerable(question)
        # answer = ''
        # if answerable:
        #     answer = self.question_answer(question)

        # return answerable, answer

        # self.base_config.logger.log_info('--------------------- 答案生成 ---------------------\n')
        answer, supporting_id_list, flag = self.question_answer_pipeline(question, useful_information, short_flag)
        self.base_config.logger.log_info(f'answer:{answer}\n supporting_id_list:{supporting_id_list}\n')
        
        if not flag:
            return False, answer, supporting_id_list
        if self.base_config.args.dataset != 'webqa':
            answer = self.simply_answer(question, answer)
        return True, answer, supporting_id_list
        
    
    def judge_answerable(self, question):
        """判断问题是否可回答"""
        self.base_config.logger.log_info('----判断问题是否可回答----\n')

        judge_answerable_prompt = AnswerablePrompt()
        prompt = judge_answerable_prompt.create_prompt(question)
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER: 
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')

        answerable = 'yes' in result.lower()
        return answerable

    def question_answer(self, question):
        """生成问题的答案"""
        self.base_config.logger.log_info('----基于整合的数据生成答案----\n')

        answer_prompt = QuestionAnswerPrompt()
        prompt = answer_prompt.create_prompt(question)
        
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        answer, supporting_id_list = extract_answer_text(result)
        return answer
    
    def question_answer_pipeline(self, question, useful_information, short_flag):
        """生成问题的答案"""
        answer_prompt = QuestionAnswerPipelinePrompt()
        prompt = answer_prompt.create_prompt(question, useful_information, short_flag)
        
        answer_dict = defaultdict(int)
        text_answer_dict = defaultdict(int)
        image_answer_dict = defaultdict(int)
        flag_one = False
        supporting_id_list_dict = {}

        def _run_inference():
            self.base_config.LLM.init_message()
            llm_result = self.base_config.LLM.predict(self.base_config.args, prompt)
            self.base_config.logger.log_info(f'User Prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')
            return llm_result

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_inference) for _ in range(3)]

            for future in futures:
                # try:
                    result, flag = future.result()
                    if flag:
                        flag_one = True
                    if flag and result and 'not enough to answer' not in result.lower():
                        answer, supporting_id_list = extract_answer_text(result)
                        # answer, supporting_id_list = extract_answer_supporting_id_list(result)
                        answer_lower = answer.lower()
                        answer_dict[answer_lower] += 1
                        if answer_lower in supporting_id_list_dict.items():
                            supporting_id_list_dict[answer_lower]+= supporting_id_list
                        else:
                            supporting_id_list_dict[answer_lower] = supporting_id_list
                        self.base_config.logger.log_info(f'answer:{answer_lower}, data_id_list:{supporting_id_list}\n')
                # except Exception as e:
                #     print(f"Inference error: {e}")
        
        self.base_config.logger.log_info(f'Answer dict:\n{answer_dict}\n')
        if answer_dict:
            if self.base_config.args.dataset == 'webqa':
                for answer, count in answer_dict.items():
                    if '_' in ','.join(supporting_id_list_dict[answer]):
                        text_answer_dict[answer] +=1
                    else:
                        image_answer_dict[answer]+=1
                
            if text_answer_dict:
                max_answer = max(text_answer_dict, key=lambda k: text_answer_dict[k])
                print(supporting_id_list_dict[max_answer])
                supporting_id_list = list(set(supporting_id_list_dict[max_answer]))
                return max_answer, supporting_id_list, True
                
            max_answer = max(answer_dict, key=lambda k: answer_dict[k])
            print(supporting_id_list_dict[max_answer])
            supporting_id_list = list(set(supporting_id_list_dict[max_answer]))
            return max_answer, supporting_id_list, True

        elif not answer_dict and flag_one:
            return 'not enough to answer', [], False
        else:
            raise ValueError('openai content error')
            
    # def question_answer_pipeline(self, question, useful_information):
    #     """生成问题的答案"""
    #     # self.base_config.logger.log_info('----基于整合的数据生成答案----\n')

    #     answer_prompt = QuestionAnswerPipelinePrompt()
    #     prompt = answer_prompt.create_prompt(question, useful_information)
        
    #     for i in range(0,3):
    #         self.base_config.LLM.init_message()
    #         result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
    #         if not flag:
    #             raise ValueError('openai content error')

        # self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
        # self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        
    
    def simply_answer(self, question, answer):
        simply_answer_prompt = SimplyAnswerPrompt()
        prompt = simply_answer_prompt.create_prompt(question, answer)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        result = extract_output(llm_result)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---简化答案---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')
        
        return result

class AnswerGenerateLittlePhase:
    def __init__(self, base_config, kg_config):
        self.base_config = base_config
        self.kg_config = kg_config

    def run(self, question, useful_information, short_flag):
        
        self.base_config.logger.log_info('--------------------- 答案生成 ---------------------\n')
        # answerable = self.judge_answerable(question)
        # answer = ''
        # if answerable:
        #     answer = self.question_answer(question)

        # return answerable, answer

        answer, flag = self.question_answer_pipeline(question, useful_information, short_flag)
        if not flag:
            return False, answer
        # if 'not enough to answer' in result.lower():
        #     return False, result
        
        if self.base_config.args.dataset != 'webqa':
            answer = self.simply_answer(question, answer)
        return True, answer
        
    
    def judge_answerable(self, question):
        """判断问题是否可回答"""
        self.base_config.logger.log_info('----判断问题是否可回答----\n')

        judge_answerable_prompt = AnswerablePrompt()
        prompt = judge_answerable_prompt.create_prompt(question)
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')

        answerable = 'yes' in result.lower()
        return answerable

    def question_answer(self, question, useful_information):
        """生成问题的答案"""
        self.base_config.logger.log_info('----基于整合的数据生成答案----\n')

        answer_prompt = QuestionAnswerLittlePrompt()
        prompt = answer_prompt.create_prompt(question, useful_information)
        
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        answer = extract_answer_text(result)
        return answer
    
    
    def question_answer_pipeline(self, question, useful_information, short_flag):
        """生成问题的答案"""
        answer_prompt = QuestionAnswerPipelinePrompt()
        prompt = answer_prompt.create_prompt(question, useful_information, short_flag)
        
        answer_dict = defaultdict(int)
        flag_one = False

        def _run_inference():
            self.base_config.LLM.init_message()
            return self.base_config.LLM.predict(self.base_config.args, prompt)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_inference) for _ in range(3)]

            for future in futures:
                try:
                    result, flag = future.result()
                    if flag:
                        flag_one = True
                    if flag and 'not enough to answer' not in result.lower():
                        flag_one = True
                        answer = extract_answer_text(result)
                        answer_lower = answer.lower()
                        answer_dict[answer_lower] += 1
                except Exception as e:
                    print(f"Inference error: {e}")

        if answer_dict:
            max_answer = max(answer_dict, key=lambda k: answer_dict[k])
            return max_answer, True
        elif not answer_dict and flag_one:
            return 'not enough to answer', False
        else:
            raise ValueError('openai content error')
    
    # def question_answer_pipeline(self, question, useful_information):
    #     """生成问题的答案"""
    #     # self.base_config.logger.log_info('----基于整合的数据生成答案----\n')

    #     answer_prompt = QuestionAnswerPipelinePrompt()
    #     prompt = answer_prompt.create_prompt(question, useful_information)
        
    #     flag_one = False
    #     answer_dict = {}
    #     for i in range(0, 3):
    #         self.base_config.LLM.init_message()
    #         result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
    #         if flag and 'not enough to answer' not in result.lower():
    #             flag_one = True
    #             answer = extract_answer_text(result)
    #             if answer.lower() not in answer_dict.keys():
    #                 answer_dict[answer.lower()] += 1
    #     if answer_dict:
    #         max_answer = max(answer_dict, key=answer_dict.get)
    #         return max_answer, True
    #     if not answer_dict and flag_one:
    #         return 'not enough to answer', False
        
    #     if not flag_one:
    #         raise ValueError('openai content error')
        
            
        # self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
        # self.base_config.logger.log_info(f'LLM response:\n{result}\n')

    def simply_answer(self, question, answer):
        simply_answer_prompt = SimplyAnswerPrompt()
        prompt = simply_answer_prompt.create_prompt(question, answer)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        result = extract_output(llm_result)
        # self.base_config.logger.log_info(f'---简化答案---\n')
        # self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
        # self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')
        
        return result
    

class AnswerValidatePhase:
    def __init__(self, base_config, kg_config):
        self.base_config = base_config
        self.kg_config = kg_config

    def run(self, question, answer, entity_information, agent_response_information, KDCI_flag=1):
        self.base_config.logger.log_info('--------------------- 答案验证 ---------------------\n')
        if answer not in self.kg_config.kg_entities_name:
            entity_information = ''
        information = entity_information + agent_response_information
        
        semantics, llm_result = self.validate_answer_semantics(question, answer, entity_information)
        if semantics:
            if not KDCI_flag:
                return True
            consistent, llm_result = self.validate_answer_information(question, answer, information)
            if consistent:
                # temp_answer = self.question_answer(question, agent_response_information)
                # if 'no answer' in temp_answer.lower():
                    # return False
                # if  temp_answer.lower() == answer.lower() or answer.lower() in temp_answer.lower() or temp_answer.lower() in answer.lower():
                return True

        return False

    def simply_answer(self, question, answer):
        simply_answer_prompt = SimplyAnswerPrompt()
        prompt = simply_answer_prompt.create_prompt(question, answer)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        result = extract_output(llm_result)
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---简化答案---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')
        
        return result

    def question_answer_with_feedback(self, question):
        prompt = f'''
Please revise the answer based on this feedback information.
Output format:
answer:<answer>
        '''
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---基于反馈生成答案---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        temp_answer = extract_output(llm_result)
        answer = self.simply_answer(question, temp_answer)
        return answer

    def question_answer(self, question, information):
        question_answer_prompt = QuestionAnswerByDataPrompt()
        prompt = question_answer_prompt.create_prompt(question, information)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---基于源数据生成答案---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        answer = extract_answer(llm_result)
        return answer
        
    def validate_answer_semantics(self, question, answer, entity_information):
        validate_semantics_prompt = ValidateSemanticsPrompt()
        prompt = validate_semantics_prompt.create_prompt(question, answer, entity_information)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---验证答案的语义是否与问题相符---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        return 'yes' in llm_result.lower(), llm_result
    
    def validate_answer_information(self, question, answer, information):
        validate_information_prompt = ValidateAnswerInformationPrompt()
        prompt = validate_information_prompt.create_prompt(question, answer, information)
        self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        result = extract_output_reason(llm_result)

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---验证答案与数据的一致性---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        return 'yes' in result.lower(), result

class AnswerGenerateAndValidatePhase:
    def __init__(self, base_config, kg_config):
        self.base_config = base_config
        self.kg_config = kg_config

    def run(self, question, answer, entity_information, agent_response_information, useful_information):
        self.base_config.logger.log_info('--------------------- 答案验证 ---------------------\n')
        if answer not in self.kg_config.kg_entities_name:
            entity_information = ''
        information = entity_information + agent_response_information
        
        semantics, consistent = False, False
        
        semantics, llm_result = self.validate_answer_semantics(question, answer, entity_information)
        if semantics:
            consistent, llm_result = self.validate_answer_information(question, answer, information)
            if consistent:
                return  True, answer
        # return False, llm_result
            else:
                next_answer = self.question_answer_with_feedback(question, useful_information)
                return False, next_answer
        else:
            next_answer = self.question_answer_with_feedback(question, useful_information)
            return False, next_answer

    def question_answer(self, question, useful_information, feedback_flag):
        if feedback_flag==1:
            return self.question_answer_with_feedback(question, userful_information)
        else:
            return self.question_answer_with_no_feedback(question, userful_information)

    def question_answer_with_feedback(self, question, userful_information, feedback):
        question_answer_with_feedback_prompt = QuestionAnswerWithFeedbackPrompt()
        prompt = question_answer_with_feedback_prompt.create_prompt(question, useful_information, feedback)
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        answer, supporting_id_list = extract_answer_text(result)
        return answer
        
    def question_answer_with_no_feedback(self, question, useful_information):
        """生成问题的答案"""
        self.base_config.logger.log_info('----基于整合的数据生成答案----\n')

        answer_prompt = QuestionAnswerLittlePrompt()
        prompt = answer_prompt.create_prompt(question, useful_information)
        self.base_config.LLM.init_message()
        result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{result}\n')
        answer, supporting_id_list = extract_answer_text(result)
        return answer
        
    def validate_answer_semantics(self, question, answer, entity_information):
        validate_semantics_prompt = ValidateSemanticsPrompt()
        prompt = validate_semantics_prompt.create_prompt(question, answer, entity_information)
        # self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---验证答案的语义是否与问题相符---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        return 'yes' in llm_result.lower(), llm_result
    
    def validate_answer_information(self, question, answer, information):
        validate_information_prompt = ValidateAnswerInformationPrompt()
        prompt = validate_information_prompt.create_prompt(question, answer, information)
        # self.base_config.LLM.init_message()
        llm_result, flag = self.base_config.LLM.predict(self.base_config.args, prompt)
        if not flag:
            raise ValueError('openai content error')
        result = extract_output_reason(llm_result)

        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'---验证答案与数据的一致性---\n')
            self.base_config.logger.log_info(f'User prompt:\n{prompt}\n')
            self.base_config.logger.log_info(f'LLM response:\n{llm_result}\n')

        return 'yes' in result.lower(), result
    
    

