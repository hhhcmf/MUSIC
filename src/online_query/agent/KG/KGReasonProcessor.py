from concurrent.futures import ThreadPoolExecutor
from KG_LLM.online_query.retrieve import RetrieveRelationsPhase
import time

class KGReasonProcessor:
    def __init__(self, base_config, kg_config, retrieve_entities_phase, kge_retrieve_phase):
        self.base_config = base_config
        self.retrieve_entities_phase = retrieve_entities_phase
        self.retrieve_relations_phase = RetrieveRelationsPhase(self.retrieve_entities_phase.retrieve_config, top_triple_k = 5)
        self.kg_config = kg_config
        self.kge_retrieve_phase = kge_retrieve_phase
    
    def get_entity_data_by_title(self, title, modality_type):
        entity_data = self.kg_config.KG.get_entity_by_title(title, modality_type, self.kg_config.data_id_list)
        return entity_data

    def get_subgraph_entity_information(self, question, data_id_list, query_entity_list, question_embedding, kge_flag = 1, KDCI_flag = 1):
        retrieved_entity_data_list = []
        entities_information = ''
        query_entities_information, kge_entities_information = '', ''

        start_time = time.time()
        query_entity_list = list(set(query_entity_list))
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda query_entity: self.QueryEntity(question, query_entity, question_embedding, KDCI_flag),
                query_entity_list
            ))

        query_entity_information_dict_list = []
        for query_entity_information_dict, entity_data_list in results:
            retrieved_entity_data_list.extend(entity_data_list)
            # query_entity_information_dict_list.append(query_entity_information_dict)

            # for key, value in query_entity_information_dict.items():
            #     entities_information += f'Related to {key}:\n{value["entity_information"]}\n'
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'retrieve time(fastext):{time.time()-start_time}')
        
        if not kge_flag:
            retrieved_entity_data_list = list(set(retrieved_entity_data_list))
            return entities_information, retrieved_entity_data_list, query_entity_information_dict_list, ''

        # 运行 KGE 召回
        start_time = time.time()
        retrieved_entity_data_list = list(set(retrieved_entity_data_list))
        subgraph_entity_data_list = self.kge_retrieve_phase.run(
            question, query_entity_list, data_id_list, retrieved_entity_data_list
        )
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f'retrieve time(kge):{time.time()-start_time}')
        
        # 处理 KG 实体信息
        if not KDCI_flag:

            # kge_entity_data_list = list(set(subgraph_entity_data_list) - set(retrieved_entity_data_list))
            kge_entity_data_list = list(set(subgraph_entity_data_list))
            start_time = time.time()
            kge_entities_information_dict_list = self.get_kge_entities_information(
                question, kge_entity_data_list, question_embedding, KDCI_flag
            )
            query_entity_information_dict_list.extend(kge_entities_information_dict_list)
            for kge_entities_information_dict in kge_entities_information_dict_list:
                for key, value in kge_entities_information_dict.items():
                    # entities_information += f'Related to {key}:\n{value["entity_information"]}\n'
                    entities_information += f'{value["entity_information"]}\n'
            # entities_information += kge_entities_information
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f'retrieve time(get_kge_entities_information):{time.time()-start_time}')
        
        
        #下面未进行并行化
        # for idx, query_entity in enumerate(query_entity_list):
        #     query_entity_information, entity_data_list = self.QueryEntity(question, query_entity, question_embedding)
        #     retrieved_entity_data_list.extend(entity_data_list)
        # subgraph_entity_data_list = self.kge_retrieve_phase.run(question, query_entity_list, data_id_list, retrieved_entity_data_list)
        
        # entities_information = self.get_kg_entities_information(question, entity_data_list, question_embedding)
            # entities_information += f'{idx}.{query_entity_information}\n'
        
        return entities_information, subgraph_entity_data_list, query_entity_information_dict_list, ''

    def get_query_entities_information(self, question, query_entity_list, question_embedding):
        retrieved_entity_data_list = []
        entities_information = '**Knowledge Graph Information:**\n'
        
        for idx, query_entity in enumerate(query_entity_list):
            query_entity_information, entity_data_list = self.QueryEntity(question, query_entity, question_embedding)
            retrieved_entity_data_list.extend(entity_data_list)
            for key, value in query_entity_information.items():
                entities_information += f'{value["entity_information"]}\n'
        
        return entities_information, list(set(retrieved_entity_data_list))
    
    def get_query_entities_information_dict(self, question, query_entity_list, question_embedding):
        retrieved_entity_data_list = []
        entity_information_dict_list = []
        for idx, query_entity in enumerate(query_entity_list):
            query_entity_information, entity_data_list = self.QueryEntity(question, query_entity, question_embedding)
            retrieved_entity_data_list.extend(entity_data_list)
            entity_information_dict_list.append(query_entity_information)
        return entity_information_dict_list, list(set(retrieved_entity_data_list))
    
    def get_entity_data_by_name(self, entity_name):
        entity_data_list = self.kg_config.KG.get_entity_by_entity(entity_name, self.kg_config.data_id_list)
        return entity_data_list

    def get_kg_entities_information(self, question, entity_data_list, question_embedding, KDCI_flag=1):
        entities_information = '**Knowledge Graph Information:**\n'
        text_entities_information = ''
        pos, text_pos = 0, 0
        for idx, entity_data in enumerate(entity_data_list):
            if not entity_data:
                continue
            # if entity_data.type == 'image' and entity_data.title != entity_data.entity_name and KDCI_flag:
            #     continue

            pos += 1
            entity_information, retrieved_entity_data_list  = self.retrieve_entities_phase.query_entity_information(entity_data, question, question_embedding, self.kg_config.entity_link_hash)
            entities_information += f'{pos}.{entity_information}\n'
            if entity_data.type == 'text':
                text_pos += 1
                text_entities_information += f'{text_pos}.{entity_information}\n'
        
        return entities_information


    def get_kge_entities_information(self, question, entity_data_list, question_embedding, KDCI_flag=1):
        entities_information = ''
        text_entities_information = ''
        pos, text_pos = 0, 0
        entities_information_dict_list = []
        for idx, entity_data in enumerate(entity_data_list):
            entities_information_dict = {}
            if not entity_data:
                continue
            
            pos += 1
            entity_information, retrieved_entity_data_list  = self.retrieve_entities_phase.query_entity_information(entity_data, question, question_embedding, self.kg_config.entity_link_hash, KDCI_flag)
            entities_information_dict[entity_data.entity_name] = {}
            entities_information += f'{pos}.{entity_information}\n'
            entities_information_dict[entity_data.entity_name]['entity_information'] = f'{entity_information}'
            # entities_information_dict[entity_data.entity_name][entity_name] = entity_information
            
            entities_information_dict[entity_data.entity_name]['entity_data_list'] = retrieved_entity_data_list
            entities_information_dict_list.append(entities_information_dict)
        return entities_information_dict_list
        

    
    def QueryEntity(self, question, query_entity, question_embedding, KDCI_flag=1):
        retrieved_entity_data_list = []

        entities_dict = self.retrieve_entities_phase.retrieve(question, query_entity)
        query_entity_information = ''
        query_entity_information_dict = {}
        query_entity_information_dict[query_entity] = {}
        
        for entity_name, related_entity_data_list in entities_dict.items():
            retrieved_entity_data_list += related_entity_data_list
            for idx, entity_data in enumerate(related_entity_data_list):
                related_entity_information, entity_data_list  = self.retrieve_entities_phase.query_entity_information(entity_data, question, question_embedding, self.kg_config.entity_link_hash, KDCI_flag)
                retrieved_entity_data_list += entity_data_list
                query_entity_information += f'{idx+1}.{related_entity_information}\n'
                query_entity_information_dict[query_entity][entity_data] = related_entity_information
        
        query_entity_information_dict[query_entity]['entity_information'] = query_entity_information
        retrieved_entity_data_list = list(set(retrieved_entity_data_list))
        query_entity_information_dict[query_entity]['entity_data_list'] = retrieved_entity_data_list
        
        
        return query_entity_information_dict, retrieved_entity_data_list
    
        
    def get_agent_response_information(self, question, question_embedding, text_entity_data_list, image_qa_list, table_records_list):
        agent_response_information = ''
        passage_information, table_record_information, image_qa_information = '', '', ''
        
        if text_entity_data_list:
            passage_information += '\n**Passages Data:**\n'
            
            for idx, (data_id, entity_name, related_snippet) in enumerate(text_entity_data_list):
                text_triples_information = ''
                text_entity_data = self.kg_config.KG.get_entity_by_IdAndEntity(data_id, entity_name)
                triples = self.kg_config.KG.get_triples_by_data_id(data_id)
                related_triples = self.retrieve_relations_phase.retrieve(question, question_embedding, triples)
                for triple in related_triples:
                    head_entity = triple.head_entity
                    tail_entity = triple.tail_entity
                    relation = triple.relation
                    description = triple.description
                    text_triples_information += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
                if not text_entity_data:
                    continue
                idx += 1
                passage_information += f'{idx}. data id:{text_entity_data.data_id}\n title:{text_entity_data.title}\n'
                # print('get_agent_response_information:', text_entity_data.data_id, text_entity_data.title, text_entity_data.entity_name)
                # print(f'text_entity_data.text:{text_entity_data.text}')
                if len(text_entity_data.text)<1000:
                    passage_information += f'passage:{text_entity_data.text}\n'
                passage_information += f'related Text Snippet:{related_snippet}\n'
                passage_information += f'related Triples:\n{text_triples_information}\n'
        else:
            passage_information += '\n**Passages Data:**\nnone\n'

        

        if image_qa_list:
            image_qa_information += '\n**Image Data:**\n'
            for idx, image_qa_details in enumerate(image_qa_list):
                data_id, title, img_question, img_answer, alias_names = image_qa_details
                alias_name_str = ','.join(alias_names)
                idx += 1
                # data_id = self.kg_config.KG.get_entity_by_title(title, 'image', self.kg_config.data_id_list).data_id
                image_qa_information += f'{idx}. data id:{data_id}\n image title:{title}\n the alias name of the image:{alias_name_str}\n image question:{img_question}\n image answer:{img_answer}\n'
        else:
            image_qa_information += '\n**Image Data:**\nnone\n'
        
        
        if table_records_list:
            table_record_information += '\n**Table Data:**\n'
        else:
            table_record_information += '\n**Table Data:**\nnone\n'
        
        if table_records_list:
            for idx, record in enumerate(table_records_list):
                # print('/KGReasonProcessor:table:', record)
                if record[0]:
                    table_name, table_desp, select_columns, conditions, output, rows = record
                    idx += 1
                    entity_data = self.kg_config.KG.get_entity_by_table_name(table_name, self.kg_config.data_id_list)
                    columns = entity_data.columns
                    if table_desp:
                        table_record_information += f'{idx}. data id:{entity_data.data_id}\ntable name:{table_name}\ntable description:{table_desp}\nselect columns:{select_columns}(columns are separated by "@#")\n table records:\n{output}\n'
                    else:
                        table_record_information += f'{idx}. data id:{entity_data.data_id}\ntable name:{table_name}\nselect columns:{select_columns}(columns are separated by "@#")\n table records:\n{output}\n'
        
        agent_response_information = passage_information + image_qa_information + table_record_information
        return agent_response_information, passage_information, image_qa_information, table_record_information

    def get_text_agent_one_information(self, text_entity_data):
        if not text_entity_data:
            return 'none'
        passage_information += f'{idx}. data id:{text_entity_data.data_id}\n title:{text_entity_data.title}\n'
        if len(text_entity_data.text)<100:
            passage_information += f'passage:{text_entity_data.text}\n'
        passage_information += f'related text snippet:{text_entity_data.related_snippet}\n'
        
        triples_information = ''
        triples = self.kg_config.KG.get_triples_by_entity(entity_name, text_entity_data.data_id)
        for triple in triples:
            head_entity = triple.head_entity
            tail_entity = triple.tail_entity
            relation = triple.relation
            description = triple.description
            triples_information += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
        passage_information += f'triples:\n{triples_information}'

        return passage_information

    def get_image_agent_one_information(self, image_qa_data):
        if not image_qa_data:
            return 'none'
        data_id, title, img_question, img_answer = image_qa_data
        # data_id = self.kg_config.KG.get_entity_by_title(title, 'image', self.kg_config.data_id_list).data_id
        image_qa_information = f'data id:{data_id}\n image title:{title}\n  image question:{img_question}\n image answer:{img_answer}\n'
        return image_qa_information

        
        #TODO
        #先获取子图，然后检索相关的，然后组织信息(表格数据可能得处理一下)，然后让生成答案，不可以继续检索相关的实体
    def get_qury_entity_information(self, question, data_id_list, query_entity_list, question_embedding, KDCI_flag = 0):
        retrieved_entity_data_list = []
        entities_information = ''
        query_entities_information, kge_entities_information = '', ''

        start_time = time.time()
        query_entity_list = list(set(query_entity_list))
        # 使用 ThreadPoolExecutor 并行执行 QueryEntity 查询
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda query_entity: self.QueryEntity(question, query_entity, question_embedding),
                query_entity_list
            ))

        # 收集查询结果
        query_entity_information_dict_list = []
        for query_entity_information_dict, entity_data_list in results:
            retrieved_entity_data_list.extend(entity_data_list)
            query_entity_information_dict_list.append(query_entity_information_dict)

            for key, value in query_entity_information_dict.items():
                entities_information += f'Related to {key}:\n{value["entity_information"]}\n'
        
        retrieved_entity_data_list = list(set(retrieved_entity_data_list))
        return entities_information, retrieved_entity_data_list, query_entity_information_dict_list, ''

    
    