from online_query.agent.base import baseAgent
from online_query.agent.text.prompt import TextPrompt
from online_query.agent.text.TextProcessor import TextProcessor
from utils.proprecss_llm_answer import extract_output
import time
class TextExtractor(baseAgent):
    def __init__(self, base_config, kg_config):   
        super().__init__(base_config)
        self.kg_config = kg_config
        self.prompt_generator = TextPrompt(kg_config)
        self.text_processor = TextProcessor()
        # self.retrieve_entities_phase = retrieve_entities_phase

    def inference(self, question, userful_information, text_entity_data_list):
        '''
        针对文本模态的推理,返回与question有关的文本实体信息,并且包含相关的文本片段,组成
        '''
        inference_all_start_time = time.time()
        self.base_config.logger.log_info("----------agent:TextExtractor被激活--------\n")
        self.base_config.logger.log_info("step 1:获取与question有关的文本实体\n")
        #step 1:获取与question有关的实体

        # 定义一个简易“分批”函数，每批最多 10 个
        def chunkify(lst, chunk_size=6):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        # 用于收集所有批次的 related_entities
        related_entities = []

        # 遍历所有批次
        inference_start_time = time.time()
        for chunk in chunkify(text_entity_data_list, 6):
            # 生成 prompt
            start_time = time.time()
            judge_text_prompt = self.prompt_generator.create_judge_prompt(
                question, userful_information, chunk
            )
            self.base_config.LLM.init_message()
            llm_result = self.base_config.LLM.predict(
                self.base_config.args, judge_text_prompt
            )
            # 从当前批次提取到的实体
            partial_related = self.text_processor.extract_entity_and_data_id(llm_result)

            # 记录日志
            self.base_config.logger.log_error(f"LLM 执行提取部分实体 time：{time.time()-start_time}")
            self.base_config.logger.log_info(f"User prompt:\n{judge_text_prompt}\n")
            self.base_config.logger.log_info(f"LLM response:\n{llm_result}\n")
            self.base_config.logger.log_info(f"Partial related_entities: {partial_related}\n")

            # 累加到最终列表
            related_entities.extend(partial_related)
        self.base_config.logger.log_error(f"LLM 执行提取全部实体 time：{time.time()-inference_start_time}")

        if not related_entities:
            return []

        #step 2:提取与question有关实体的文本的相关文本片段
        
        self.base_config.logger.log_info("step 2:提取与question有关实体的文本的相关文本片段\n")
        text_entity_data_list = []
        # if related_entities.lower == 'no':
        #     return []
        
        for entity_name, data_id in related_entities:
            # print('text inference:', entity_name, data_id)
            text_entity_data = self.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name) #这个检索会导致text_entity_data的related_snippet肯定为空
            if not text_entity_data or text_entity_data.type!='text':
                continue
            text = text_entity_data.text
            title = text_entity_data.title
            start_time = time.time()
            related_snippet= self.ExtractText(userful_information, question, title, text)
            text_entity_data.related_snippet = related_snippet
            text_entity_data_list.append((text_entity_data.data_id, text_entity_data.entity_name, related_snippet))
        self.base_config.logger.log_error(f"LLM 执行提取文本 time：{time.time()-start_time}")
        self.base_config.logger.log_info(f'inference(text) time:{time.time()-inference_all_start_time}')
        return text_entity_data_list
            
    
    def ExtractText(self, userful_information, question, title, text):

        reasoning_prompt =  self.prompt_generator.create_reasoning_prompt(userful_information, question, title, text)
        self.base_config.LLM.init_message()
        llm_result = self.base_config.LLM.predict(self.base_config.args, reasoning_prompt)
        related_snippet = extract_output(llm_result)

        self.base_config.logger.log_info(f"User prompt:{reasoning_prompt}\n")
        self.base_config.logger.log_info(f"LLM response:{llm_result}\n")
        self.base_config.logger.log_info(f"Related_snippet:{related_snippet}\n")

        return related_snippet

