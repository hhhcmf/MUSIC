import sqlite3
from KG_LLM.phase.base import EmbedModel, Config, BaseConfig

class EntityInformationRetrievalPhase:
    def __init__(self, base_config, kg_config, embed_model, retrieve_config, else_config):
        self.base_config = base_config
        self.kg_config = kg_config
        self.embed_model = embed_model
        self.retrieve_config = retrieve_config
        self.else_config = else_config

    def retrieve_entity_information(self, entities, question, question_embedding):
        image_title_list, information, table_information = [], '', ''
        table_entity_data_list, image_entity_data_list, text_entity_data_list = [], [], []
        query_entity_api_answered_list, query_table_api_answered_list, image_qa_api_answered_list = {}, {}, {}
        query_entity_apis, query_table_apis = '', ''
        query_entity_information, query_table_information, image_qa_information = '', ''
        # Step 1: Retrieve information for each entity
        for entity in entities:
            entity_info = self._query_entity(entity, question, question_embedding)
            query_entity_api_answered_list[entity] = entity_info['output']
            if entity_info['flag']:
                self._append_entity_data(entity_info, image_title_list, table_entity_data_list, image_entity_data_list, text_entity_data_list)

        # Step 2: Query table data if available
        if query_entity_api_answered_list:
            self._query_table_data(table_entity_data_list, query_table_api_answered_list)
            information, table_information, query_table_information, query_table_apis = self._format_table_information(table_data, query_table_api_answered_list, information, table_information)

        # Step 3: Image QA if image titles are available
        if image_title_list:
            image_answer, image_qa_api_answered_list, image_qa_information, flag = self.origin_image_qa(question, image_title_list)

        # Step 4: Format and return the output
        information, query_entity_information, query_entity_apis = self._format_entity_output(query_entity_api_answered_list)
        return (image_title_list, information, table_information, query_entity_api_answered_list, 
                query_table_api_answered_list, image_qa_api_answered_list, query_entity_apis, query_table_apis, 
                table_entity_data_list, image_entity_data_list, 
                text_entity_data_list, query_entity_information, query_table_information, image_qa_information)

    def origin_image_qa(self, question, image_title_list):
        image_qa_prompt = ImageQAPrompt()
        prompt = image_qa_prompt.create_prompt(question, image_title_list)
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        
        if 'no need' in result.lower():
            return '', {}, '', False

        post_result, flag = extract_reasoning(result)
        self.base_config.logger.log_info(f'LLM response for image QA:\n{result}')
        image_qa_api_answered_list = {}
        image_qa_information = ''
        
        while not isinstance(post_result, dict):
            result = self._handle_format_error(prompt)
            self.base_config.logger.log_info('----------origin_image_qa---------------')
            self.base_config.logger.log_info(f'user prompt:\n{prompt}')
            self.base_config.logger.log_info(f'LLM respond:\n{result}')
            post_result, flag = extract_reasoning(result)

        for pos, (key, value) in enumerate(post_result.items(), start=1):
            title = value.get('Input', {}).get('title', '').upper()
            img_question = value.get('Input', {}).get('question')
            api_output, flag = ImageQA(self.kg_config.KG, title, img_question, self.else_config.img_folder)
            image_qa_api_answered_list[(title, img_question)] = api_output 
            image_qa_information += f'{pos}.API: ImageQA Input: {{"title":"{title}", "question": "{img_question}"}} output:\n{api_output}'
        self.base_config.logger.log_info(f'image_qa respond:\n{api_output}')
        
        return api_output, image_qa_api_answered_list, image_qa_information, True

    def _handle_format_error(self, prompt):
        """Handle cases where the output format is incorrect by prompting LLM to follow the correct format."""
        prompt += "\nOutput format error. Please follow the provided output format:\n" + '''
{
"API Access": {
    "API": "<Interface name>",
    "Input": <JSON-formatted input>
}
}
        '''
        result = self.base_config.LLM.predict(self.base_config.args, prompt)
        self.base_config.logger.log_info(f'LLM response for format error correction:\n{result}')
        return result

    def _query_entity(self, entity, question, question_embedding):
        entity = entity.upper()
        table_entity_data, image_entity_data, text_entity_data, output, flag = QueryEntity(
            entity, question, question_embedding, self.kg_config.entity_link_hash, self.base_config.LLM, 
            self.kg_config.KG, self.kg_config.kg_entities_name, self.base_config.args, self.base_config.logger, self.base_config.logger, 
            self.base_config.args.embedding_model_name, self.embed_model.model, self.embed_model.tokenizer, 
            self.retrieve_config.entity_index, self.retrieve_config.entity_id_to_key, self.retrieve_config.entity_desp_index, 
            self.retrieve_config.entity_desp_id_to_key, self.retrieve_config.triple_index, self.retrieve_config.triple_id_to_key, 
            self.retrieve_config.triple_desp_index, self.retrieve_config.triple_desp_id_to_key, 
            self.retrieve_config.top_entity_k, self.retrieve_config.top_triple_k
        )
        return {
            'table_entity_data': table_entity_data,
            'image_entity_data': image_entity_data,
            'text_entity_data': text_entity_data,
            'output': output,
            'flag': flag
        }

    def _append_entity_data(self, entity_info, image_title_list, table_entity_data_list, image_entity_data_list, text_entity_data_list):
        table_entity_data_list += entity_info['table_entity_data']
        image_entity_data_list += entity_info['image_entity_data']
        text_entity_data_list += entity_info['text_entity_data']
        for image_entity in entity_info['image_entity_data']:
            if image_entity.isImageTitle:
                image_title_list.append(image_entity.title)
        

    def _query_table_data(self, table_entity_data_list, query_table_api_answered_list):
        """Perform SQL query on tables related to entity data."""
        conditions, table_data = {}, []
        for table_entity_data in table_entity_data_list:
            columns = table_entity_data.columns.replace('@#', ',')
            table_name = table_entity_data.table_name
            if table_entity_data.is_column_value:
                conditions[table_entity_data.entity_name] = table_entity_data.column
        try:
            conn = sqlite3.connect(self.else_config.db_path)
            cursor = conn.cursor()
            table_data = QueryTable(table_name, columns, conditions, cursor)
        finally:
            cursor.close()
            conn.close()
        query_table_api_answered_list[(table_name, columns, columns, tuple(conditions.items()))] = api_output

    def _format_entity_output(self, query_entity_api_answered_list):
        """Format the entity query results for the output."""
        information, query_entity_information, query_entity_apis = '', '', ''
        for pos, (entity_name, output) in enumerate(query_entity_api_answered_list.items(), start=1):
            information += f'{pos}.{output}\n'
            query_entity_information += f'{pos}.{output}\n'
            query_entity_apis += f'{pos}.API: QueryEntity Input:{{entity_name:"{entity_name}"}}\n'
        return information, query_entity_information, query_entity_apis
    
    def _format_table_information(self, query_table_api_answered_list, information, table_information):
        """Format and append table data into the information strings."""
        pos, query_table_pos = 0, 0
        for key, output in query_table_api_answered_list.items():
            pos += 1
            query_table_pos += 1
            table_name, select_column, columns, conditions = key
            conditions_str = ', '.join([f'"{k}:{v}"' for k, v in conditions.items()])
            information += f'{pos}.API: QueryTable Input: {{"table name": "{table_name}", "columns": "{columns}", "conditions": {{{conditions_str}}} "select column": "{select_column}"}} output:\n{output}'
            table_information += f'table name:{table_name}\ncolumn names of the table:{columns}\nconditions:{conditions_str}\n'
            query_table_information +=  f'{query_table_pos}.API: QueryTable Input: {{"table name": "{table_name}", "columns": "{columns}", "conditions": {{{conditions_str}}} "select column": "{select_column}"}} output:\n'
            query_table_information += output
            query_table_apis += f'{query_table_pos}.API: QueryTable Input: {{"table name": "{table_name}", "columns": "{columns}", "conditions": {{{conditions_str}}} "select column": "{select_column}"}}\n'
        return information, table_information, query_table_information, query_table_apis

# config = Config(db_path="path_to_db", img_folder="path_to_images", entity_link_hash=..., kg_entities_name=..., 
#                 entity_index=..., entity_id_to_key=..., entity_desp_index=..., entity_desp_id_to_key=..., 
#                 triple_index=..., triple_id_to_key=..., triple_desp_index=..., triple_desp_id_to_key=...)

# embed_model = EmbedModel(model=..., tokenizer=...)

# phase = EntityInformationRetrievalPhase(LLM=..., KG=..., args=..., logger=..., embed_model=embed_model, config=config)

# result = phase.retrieve_entity_information(entities, question, question_embedding)
