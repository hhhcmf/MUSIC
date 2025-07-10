# from prompt.prompt_generator import RelatedEntitiesPrompt, RelatedTriplesPrompt, QuestionAnswerabilityPrompt, RelatedRelationsPrompt
from utils.proprecss_llm_answer import extract_related_entities_new, extract_related_relations, extract_related_triples, extract_answerable
from KG_LLM.offline_process.kg import *
from collections import defaultdict
from KG_LLM.vector_store.lancedb import get_all_elements_similarity_by_EntityName, get_all_elements_similarity_by_relation,get_data_items_by_data_id
class RetrieveBasePhase:
    def __init__(self, retrieve_config):
        self.retrieve_config = retrieve_config
        
    def retrieve(self, *args, **kwargs):
        return self._retrieve(*args, **kwargs)

    def _retrieve(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement retrieve method.")

class RetrieveEntitiesPhase(RetrieveBasePhase):
    def __init__(self, retrieve_config, query_entity_name_embedder):
        super().__init__(retrieve_config)
        self.query_entity_name_embedder = query_entity_name_embedder
        # self.prompt_generator = RelatedEntitiesPrompt(top_entity_k)
        self.entity_call_count = {}
    
    def _calculate_k(self, query_entity):
        call_count = self.entity_call_count[query_entity]

        if call_count == 1:
            return 3
        elif call_count == 2:
            return 7
        elif call_count == 3:
            return 10
        else:
            # 从第 4 次调用开始，每次递增 3
            return 10 + (call_count - 3) * 3

    def get_entity_information(self, entity_data, question, question_embedding):
        # 将实体信息组织成一定的格式

        retrieve_relations_phase = RetrieveRelationsPhase(self.retrieve_config)
        data_id = entity_data.data_id
        data_type = entity_data.type
        title = entity_data.title
        is_column = entity_data.is_column
        is_column_value = entity_data.is_column_value
        isImageTitle = entity_data.isImageTitle
        table_name = entity_data.table_name
        columns = entity_data.columns
        column = entity_data.column
        description = entity_data.description
        text = entity_data.text
        entity_name = entity_data.entity_name
        
        result = ''
        if data_type == 'table':
            name = table_name[len(title)+1:]
            result += f'Data Id: {data_id}\n'
            result += 'Modality Type: table\n'

            table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
            
            if is_column_value:  
                result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"
            elif is_column: 
                result += f'Data Type: Column name\n'           
            elif entity_name == title:
                result += f'Data Type: Table Title\n'
            else:
                result += f'Data Type: Table Name\n'
            result += table_desp
            
        elif data_type == 'image':
            result += f'Data Id: {data_id}\n'

            result += 'Modality Type: image\n'
            if isImageTitle:
                result += f'Data Type: Image Title\n'
            else: 
                result += f'Date Type: Object\n'
            result += f'Image Title:{title}\n'
            if not isImageTitle:
                result += f"Triples: ({title}, HAS, {entity_name})\n"
            result += f'Image Description:{text}\n'
        else:

            result += f'Data Id: {data_id}\n'
            result += 'Modality Type: text\n'
            result += f'Entity Description: {description}\n'
            if len(text) < 100:
                result += f'Text Data: {text}\n'
            if entity_data.related_snippet:
                result += f'Snippet Related To Question: {related_snippet}\n'
            result += f'Triples:\n'
            # triples = self.retrieve_config.kg_config.KG.get_triples_by_entity(entity_name, self.retrieve_config.kg_config.data_id_list)
            triples = self.retrieve_config.kg_config.KG.get_triples_by_entity(entity_name, data_id)
            if len(triples)>5:
                triples = retrieve_relations_phase.retrieve(question, question_embedding, triples)
            
            for triple in triples:
                head_entity = triple.head_entity
                tail_entity = triple.tail_entity
                relation = triple.relation
                description = triple.description
                result += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
        return result

    def query_entity_information(self, entity_data, question, question_embedding, entity_link_hash, KDCI_flag = 1):
        retrieved_entity_data_list = [entity_data] # 记录获取这个entity信息的时候还会检索到的实体

        retrieve_relations_phase = RetrieveRelationsPhase(self.retrieve_config)
        
        entity_link_keys = entity_link_hash.keys()
        entity_name = entity_data.entity_name

        idx = 1
        # result = f'{idx}.'
        result = ''
        result += f'Entity Name: {entity_name}\n'
        data_id = entity_data.data_id
        data_type = entity_data.type
        title = entity_data.title
        is_column = entity_data.is_column
        is_column_value = entity_data.is_column_value
        isImageTitle = entity_data.isImageTitle
        table_name = entity_data.table_name
        columns = entity_data.columns
        column = entity_data.column
        description = entity_data.description
        text = entity_data.text
        
        pos = 0 
        if data_type == 'table':
            name = table_name[len(title)+1:]
            result += f'Data Id: {data_id}\n'
            result += 'Modality Type: table\n'

            table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
            
            if is_column_value:  
                result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"
            elif is_column:  # If it's a column name
                result += f'Data Type: Column name\n'
                
                # if alias and alias!=entity_name:
                #     alias_data_list = KG.get_entity_by_entity(alias)
                #     result += f' Its alias is "{alias}".'
                #     for alias_data in alias_data_list:
                #         result += f' The alias "{alias}" belongs to {alias_data.type} modality data.'
            
            elif entity_name == title:  # General table information
                result += f'Data Type: Table Title\n'
            else:
                result += f'Data Type: Table Name\n'
            if not KDCI_flag:
                result += f'Entity Description: {description}\n'
            result += table_desp
            if (entity_name, data_id) in entity_link_keys:
                count = 0
                for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                    retrieved_entity_data_list.append(entity_link_data)
                    result += 'Alias Entities:\n'
                    count += 1
                    result += f'  {count})\n'
                    result += f'Entity Name: {entity_link_data.entity_name}\n'
                    result += self.get_entity_information(entity_link_data, question, question_embedding)
            
        elif data_type == 'image':
            
            result += f'Data Id: {data_id}\n'
            # result += f'Entity Name: {entity_name}\n'
            result += 'Modality Type: image\n'
            if isImageTitle:
                result += f'Data Type: Image Title\n'
            else: 
                result += f'Date Type: Object\n'
            result += f'Image Title:{title}\n'
            if not isImageTitle:
                result += f"Triples: ({title}, HAS, {entity_name})\n"
            result += f'Image Description:{text}\n'
            if (entity_name, data_id) in entity_link_keys:
                count = 0
                for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                    retrieved_entity_data_list.append(entity_link_data)
                    result += 'Alias Entities:\n'
                    count += 1
                    result += f'  {count})\n'
                    result += f'Entity Name: {entity_link_data.entity_name}\n'
                    result += self.get_entity_information(entity_link_data, question, question_embedding)
        else:
           
            result += f'Data Id: {data_id}\n'
            # result += f'Entity Name: {entity_name}\n'
            result += 'Modality Type: text\n'
            result += f'Entity Description: {description}\n'
            if len(text) < 100:
                result += f'Text Data: {text}\n'
            if entity_data.related_snippet:
                result += f'Snippet Related To Question: {related_snippet}\n'
            result += f'Triples:\n'
            # result += f'{entity_name} belongs to the text modality data; the triples format is (entity, relation, entity):<description of triples>. All related triples are:'
            # triples = self.retrieve_config.kg_config.KG.get_triples_by_entity(entity_name, self.retrieve_config.kg_config.data_id_list)
            triples = self.retrieve_config.kg_config.KG.get_triples_by_entity(entity_name, data_id)
            if len(triples)>5:
                triples = retrieve_relations_phase.retrieve(question, question_embedding, triples)
            
            for triple in triples:
                head_entity = triple.head_entity
                tail_entity = triple.tail_entity
                relation = triple.relation
                description = triple.description
                result += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
            if (entity_name, data_id) in entity_link_keys:
                count = 0
                for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                    retrieved_entity_data_list.append(entity_link_data)
                    result += 'Alias Entities:\n'
                    count += 1
                    result += f'  {count})\n'
                    result += f'Entity Name: {entity_link_data.entity_name}\n'
                    result += self.get_entity_information(entity_link_data, question, question_embedding)

        return result, list(set(retrieved_entity_data_list))

    
    def _retrieve(self, question, query_entity):
        # self.retrieve_config.retrieve_logger.log_info(f'\n与{query_entity}相关的实体:\ndb_path:{self.retrieve_config.kg_config.db_path}\nall:{self.retrieve_config.kg_config.kg_entities_name}')
        self.retrieve_config.retrieve_logger.log_info(f'\n与{query_entity}相关的实体:\n')
        self.entity_call_count[query_entity] = self.entity_call_count.get(query_entity, 0) + 1
        k = self._calculate_k(query_entity)

        entities_dict = defaultdict(list)

        query_entity_data_list = []
        if query_entity in self.retrieve_config.kg_config.kg_entities_name:
            query_entity_data_list = self.retrieve_config.kg_config.KG.get_entity_by_entity(query_entity, self.retrieve_config.kg_config.data_id_list)
            # self.retrieve_config.retrieve_logger.log_info(f'query_entity_data_list:{len(query_entity_data_list)}')
            # self.retrieve_config.retrieve_logger.log_info(f'set query_entity_data_list:{len(query_entity_data_list)}')

        query_entity_embedding = self.query_entity_name_embedder([query_entity])[0]
        # print(len(query_entity_embedding))
        entity_score_results = self.retrieve_config.entity_collection.similarity_search_with_metric_all(query_entity_embedding)
        data_list = get_data_items_by_data_id(self.retrieve_config.kg_config.db_path, self.retrieve_config.kg_config.data_id_list)
        results = get_all_elements_similarity_by_EntityName(data_list, entity_score_results, k)
        result_entities = []

        if len(query_entity_data_list)>k:
            for entity_data in query_entity_data_list:
                entities_dict[entity_data.entity_name].append(entity_data)
                self.retrieve_config.retrieve_logger.log_info(f'entity_id:{entity_data.entity_id} entity name:{entity_data.entity_name} score:1  data id:{entity_data.data_id} type:{entity_data.type}\n')
        else:
            related_entity_data_list = []
            # print(list(results.items())[:-1])
            for entity_id, value in list(results.items())[:-1]:
                related_entity = self.retrieve_config.kg_config.KG.get_entity_by_entityId(entity_id)
                if related_entity not in related_entity_data_list:
                    self.retrieve_config.retrieve_logger.log_info(f'entity_id:{entity_id} entity name:{related_entity.entity_name} score:{value["score"]} data id:{related_entity.data_id} type:{related_entity.type}\n')
                    entities_dict[related_entity.entity_name].append(related_entity)
            
            last_entity_id = list(results.keys())[-1]
            last_related_entity = self.retrieve_config.kg_config.KG.get_entity_by_entityId(entity_id)
            entity_name = last_related_entity.entity_name
            last_related_entity_list = self.retrieve_config.kg_config.KG.get_entity_by_entity(entity_name, self.retrieve_config.kg_config.data_id_list)
            entities_dict[entity_name]=last_related_entity_list
            for entity_data in last_related_entity_list:
                self.retrieve_config.retrieve_logger.log_info(f'entity_id:{entity_data.entity_id} entity name:{entity_data.entity_name} score:{results[last_entity_id]["score"]} data id:{entity_data.data_id} type:{entity_data.type}\n')
        
        entities_dict = dict(entities_dict)
        
        return entities_dict
       
           
    

class RetrieveRelationsPhase(RetrieveBasePhase):
    def __init__(self, retrieve_config, top_triple_k=3):
        super().__init__(retrieve_config)
        self.top_triple_k = top_triple_k
        
    def _retrieve(self, question, question_embedding, relations):
        data_list = []
        for relation in relations:
            data_list.append({"id":str(relation.relation_id)})
        
        relationship_score_results = self.retrieve_config.relationship_collection.similarity_search_with_metric_all(question_embedding)
        results = get_all_elements_similarity_by_relation(data_list, relationship_score_results, self.top_triple_k)
        
        result_relations = [self.retrieve_config.kg_config.KG.get_relation_by_id(relation_id) for relation_id, value in results.items()]
        # self.LLM.init_message()
        # related_relations = {}
        # if len(relations) <= self.k and relations:
        #     related_relations = relations
        #     self.logger.log_info(f" direct most relavant relations:{related_relations}\n")
        # else:
        #     prompt = self.prompt_generator.create_prompt(question, relations)
        #     while not related_relations:
        #         try:
        #             self.LLM.init_message()
        #             relations = self.LLM.predict(self.args, prompt)
        #             related_relations = extract_related_relations(relations)
        #         except Exception as e:
        #             related_relations = {}
        #     self.logger.log_info(f"\n------get_related_relation------\n{prompt}\n\n")
        #     self.logger.log_info(f" llm result:{related_relations}\n")

        # result_relations = []
        # kg_triples_desp = []
        # kg_triples = []
        # kg_relations = self.KG.get_kg_relationships()
        # for relation_data in kg_relations:
        #     kg_triples_desp.append(relation_data.description)
        #     kg_triples.append(f'{relation_data.head_entity} {relation_data.relation} {relation_data.tail_entity}')
        # for head_entity, value in related_relations.items():
        #     for tail_entity, data in value.items():
        #         description = data.get('Description', '')
        #         relation = data.get('Relation', '')
        #         relation_data = self.KG.get_relation_by_triple(head_entity, relation, tail_entity)
        #         if not relation_data:
        #             triple_desp_embedding = get_embeddings(description, self.embedding_model_name, self.embed_model, self.embed_tokenizer)
        #             triple_embedding = get_embeddings(f'{head_entity} {relation} {tail_entity}', self.embedding_model_name, self.embed_model, self.embed_tokenizer)
        #             triple_desp_sort = sort_candidates(triple_desp_embedding, self.triple_desp_index, kg_triples_desp, self.triple_desp_id_to_key, self.search_type)
        #             triple_sort = sort_candidates(triple_embedding, self.triple_index, kg_triples, self.triple_id_to_key, self.search_type)

        #             triple_data_sort = {}
        #             for relation_data in kg_relations:
        #                 triple = f'{relation_data.head_entity} {relation_data.relation} {relation_data.tail_entity}'
        #                 triple_description = relation_data.description
        #                 triple_sort_index = triple_sort[triple] + triple_desp_sort[triple_description]
        #                 triple_data_sort[relation_data] = triple_sort_index
        
        #             sorted_entities = sorted(triple_data_sort.items(), key=lambda item: item[1], reverse=True)
        #             top_k = 1
        #             relations = [relation for relation, index in sorted_entities[:top_k]]
        #             result_relations += relations

        #         else:
        #             result_relations+=relation_data
        # self.logger.log_info(f" final result:\n{result_relations}\n")

        return result_relations
                        
