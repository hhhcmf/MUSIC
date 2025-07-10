class EmbedModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

class KgConfig:
    def __init__(self, KG, db_path, kg_entities_name, entity_link_hash, data_id_list):
        self.KG = KG
        self.db_path = db_path
        self.kg_entities_name = kg_entities_name
        self.entity_link_hash = entity_link_hash
        self.data_id_list = data_id_list

class ElseCofig:
    def __init__(self, db_path, img_folder):
        self.db_path = db_path
        self.img_folder = img_folder

# class RetrieveConfig:
#     def __init__(self, entity_index, entity_id_to_key, entity_desp_index, entity_desp_id_to_key,
#                  triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key,
#                  top_entity_k=5, top_triple_k=5):
#         self.entity_index = entity_index
#         self.entity_id_to_key = entity_id_to_key
#         self.entity_desp_index = entity_desp_index
#         self.entity_desp_id_to_key = entity_desp_id_to_key
#         self.triple_index = triple_index
#         self.triple_id_to_key = triple_id_to_key
#         self.triple_desp_index = triple_desp_index
#         self.triple_desp_id_to_key = triple_desp_id_to_key
#         self.top_entity_k = top_entity_k
#         self.top_triple_k = top_triple_k
class RetrieveConfig:
    def __init__(self, logger, base_config, kg_config, collections):
        self.retrieve_logger = logger
        self.base_config = base_config
        self.kg_config = kg_config
        self.collections = collections
        self.entity_collection = collections['entity']
        self.relationship_collection = collections['relation_description']

class BaseConfig:
    def __init__(self, llm_list ,logger, detail_logger, args):
        self.LLM = llm_list[0]
        self.logger = logger
        self.detail_logger = detail_logger
        self.args = args
        self.image_llm = llm_list[2]
        self.text_llm = llm_list[3]
        self.table_llm = llm_list[1]

class BasePhase:
    def __init__(self, LLM, logger, pos, args, prompt_generator):
        self.LLM = LLM
        self.logger = logger
        self.pos = pos
        self.args = args
        self.prompt_generator = prompt_generator

    def run(self):
        # 基本的 run 操作，可以在子类中重写
        self.logger.info(f"Running phase at position {self.pos}")
        # 调用 prompt_generator 生成提示
        prompt = self.prompt_generator.generate_prompt(self.args)
        # 使用 LLM 处理提示
        response = self.LLM.process(prompt)
        return response
