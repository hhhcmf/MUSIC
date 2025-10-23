import json
import sqlite3
from utils.utils import read_jsonl_file
from config.config import config_loader
from KG_LLM.offline_process.kg import KnowledgeGraph
import os

def safe_upper(value):
    return value.upper() if value else value

class MMQADataProcessor:
    def __init__(self):
        self.text_path = config_loader.get_text_path('MMQA')
        self.table_path = config_loader.get_table_path('MMQA')
        self.image_path = config_loader.get_image_path('MMQA')
        self.final_image_path = config_loader.get_final_image('MMQA')
        self.dev_path = config_loader.get_dev_path('MMQA')
        self.db_path = 'MMQA.db'
        self.kg_path = '/home/cmf/multiQA/KG_LLM/kg.db'
    
    def create_text_mapping(self, file_path=None):
        file_path = file_path or self.text_path
        data_list = read_jsonl_file(file_path)
        id_mapping = {}
        for data in data_list:
            text_id = data.get('id')
            title = data.get('title')
            text = data.get('text')
            url = data.get('url')
            if text_id:
                id_mapping[text_id] = {'id': text_id, 'title': title, 'text': text, 'url': url, 'type': 'text'}
        return id_mapping
    
    def get_title_text_by_id(self, text_id, file_path=None):
        file_path = file_path or self.text_path
        id_mapping = self.create_text_mapping(file_path)
        return id_mapping.get(text_id, {'id': None, "title": None, "text": None})

    def get_image_mapping(self, file_path=None):
        file_path = file_path or self.image_path
        data_list = read_jsonl_file(file_path)
        id_mapping = {}
        for data in data_list:
            image_id = data.get('id')
            title = data.get('title')
            path = data.get('path')
            url = data.get('url')
            if image_id:
                id_mapping[image_id] = {'id': image_id, 'title': title, 'url': url, 'path': self.final_image_path + path, 'type': 'image'}
        return id_mapping

    def get_image_by_id(self, image_id, file_path=None):
        file_path = file_path or self.image_path
        id_mapping = self.get_image_mapping(file_path)
        return id_mapping.get(image_id, {'id': None, "title": None, "path": None})
    
    def get_question_table_id(self, file_path=None):
        file_path = file_path or self.dev_path
        
        table_id_list = []
        data_list = read_jsonl_file(file_path)
        for data in data_list:
            extracted_info = self.extract_question_information(data)
            if extracted_info["table_id"]!='':
                table_id_list.append(extracted_info['table_id'])
        
        return table_id_list

    def extract_question_information(self, data):
        extracted_info = {}
        extracted_info["question"] = data.get("question", "")
        extracted_info["qid"] = data.get("qid", "")
        answers = data.get("answers", [])
        extracted_info["answer"] = []
        if answers:
            for answer in answers:
                if  answer.get('answer') != '':
                    extracted_info["answer"].append(answer.get('answer'))
        else:
            extracted_info["answer"] = []

        metadata = data.get("metadata", {})
        extracted_info["question_type"] = metadata.get("type", '')
        extracted_info["image_doc_ids"] = metadata.get("image_doc_ids", [])
        extracted_info["text_doc_ids"] = metadata.get("text_doc_ids", [])
        extracted_info["modalities"] = metadata.get("modalities", [])
        extracted_info["table_id"] = metadata.get("table_id", "")
        extracted_info['supporting_context'] = data.get('supporting_context')
        return extracted_info
        

    def get_dev_information(self, file_path = None):
        file_path = file_path or self.dev_path
        
        result = []
        text_q = []
        table_q = []
        image_q = []
        data_list = read_jsonl_file(file_path)
        for data in data_list:
            extracted_info = self.extract_question_information(data)
            result.append(extracted_info)
            if extracted_info['modalities'] == ["text"]:
                text_q.append(extracted_info)
            if extracted_info['modalities'] == ["table"]:
                table_q.append(extracted_info)
            if extracted_info['modalities'] == ["image"]:
                image_q.append(extracted_info)
        return result, text_q, table_q, image_q

    def get_supporting_context(self, file_path = None):
        file_path = file_path or self.dev_path
        question_list = self.get_dev_information(file_path)
        supporting_id_list = []
        for question in question_list:
            supporting_context = question['supporting_context']
            supporting_id = []
            for supporting in supporting_context:
                supporting_id += [supporting['doc_id']]
                supporting_id_list.append(supporting_id)
        
        return supporting_id_list

class ImageProcessor:
    def __init__(self):
        self.image_path = '/data/cmf/mmqa/data/final_dataset_images'
        self.data_processor = MMQADataProcessor()
        self.imgae_caption_path = '/data/cmf/mmqa/data/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images_with_caption.jsonl'
        self.object_path = '/data/cmf/mmqa/image_object'

    def create_image_mapping(self):
        questions, image_q, table_q, image_q = self.data_processor.get_dev_information()
        image_list = []
        for question in questions:
            image_list += question["image_doc_ids"]
        image_list = list(set(image_list))

        data_list = read_jsonl_file(self.imgae_caption_path)
        id_mapping = {}
        for data in data_list:
            image_id = data.get('id')
            title = data.get('title')
            path = data.get('path')
            url = data.get('url')
            caption = data.get('caption')
            if image_id in image_list:
                object_dict = self.get_image_object(image_id, 0.2, 0.2)
                id_mapping[image_id] = {'id': image_id, 'title': title, 'url': url, 'path': os.path.join(self.image_path, path), 'type': 'image', 'caption': caption, 'object':object_dict}
        return id_mapping

    def get_image_object(self, image_id, entity_conf_value, attribute_conf_value):
        file_path = os.path.join(self.object_path, f'{image_id}.txt' )
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        i = 0
        result_dict  ={}
        while i < len(lines):
            label_line = lines[i].strip()
            i += 1
            attributes_score_line = lines[i].strip()
            i += 1
            confidence_line = lines[i].strip()
            i += 1
            rect_line = lines[i].strip()
            i += 2

            entity = label_line.split("Label:")[1].split(' ')[2]
            attributes = label_line.split("Label:")[1].split(' ')[1]
            attribute_list = attributes.split(',')

            attributes_score_content = attributes_score_line.split("Attribute Scores:")[1].strip()
            if attributes_score_content:
                attribute_scores = list(map(float, attributes_score_content.split(',')))
            else:
                attribute_scores = []

            # 提取置信度
            entity_confidence = float(confidence_line.split("Confidence Score:")[1].strip())

            if entity_confidence > entity_conf_value:
                filtered_attributes = [attr for attr, conf in zip(attribute_list, attribute_scores) if conf > attribute_conf_value]
                if entity in result_dict:
                    result_dict[entity].extend(filtered_attributes)
                else:
                    result_dict[entity] = filtered_attributes
        return result_dict

class TableProcessor:
    def __init__(self):
        self.table_path = config_loader.get_table_path('MMQA')
        self.db_path = 'MMQA.db'
        self.data_processor = MMQADataProcessor()

    def get_table_mapping(self, file_path=None):
        file_path = file_path or self.table_path
        data_list = read_jsonl_file(file_path)
        id_mapping = {}
        for data in data_list:
            table_id = data.get('id')
            title = data.get('title')
            table = data.get('table')
            rows = table.get('table_rows')
            columns = table.get('header')
            name = table.get('table_name')
            if table_id:
                id_mapping[table_id] = {'id': table_id, 'title': title, 'table': table, 'name': name, 'rows': rows, 'columns': columns, 'type': 'table'}
        return id_mapping
    
    def get_table_by_id(self, id_mapping, table_id):
        return id_mapping.get(table_id, {'id': None,"title": None, 'name':None, 'rows': None, 'columns':None})

    
    def get_table_columns(self, table_information):
        """
        获取表格中的列信息
        输入:table_information
        输出:columns_info列表(column_name, column_type)、主键列
        """
        columns_info = []
        columns = table_information['columns']
        key_column = []
        idx = 0
        for column in columns:
            column_name = column.get("column_name")
            metadata = column.get("metadata", {})
            column_type = metadata.get("type", "Unknown")
            if metadata.get('is_key_column'):
                key_column.append((column_name, idx))
            columns_info.append((column_name, column_type))
            idx += 1
        return columns_info, key_column

    def extract_column_values_by_id(self, id_mapping, table_id, file_path=None):
        columns_info = []

        data = self.get_table_by_id(id_mapping, table_id)
        headers = [col.get("column_name") for col in data.get('columns', [])]
        table_rows = data.get('rows', [])

        for row in table_rows:
            row_data = {}
            for header, cell in zip(headers, row):
                row_data[header] = cell.get("text")
            columns_info.append(row_data)
        return columns_info

    def extract_column(self, table_information):
        headers = [col.get("column_name") for col in table_information['columns']]
        return headers

    def extract_rows(self, table_information):
        rows = []
        table_rows = table_information['rows']
        for row in table_rows:
            rows.append([item['text'] for item in row])
        return rows

    def extract_column_rows(self, table_information):
        """
        获取表格中的列和行信息
        输出：columns_info a list of row data
            row_data = {
                <column_name>: {
                    'value': <cell text>
                    'alias': <alias>
                }
            }
        """
        columns_info = []
        headers = [col.get("column_name") for col in table_information['columns']]
        table_rows = table_information['rows']

        for row in table_rows:
            row_data = {}
            for header, cell in zip(headers, row):
                row_data[header] = {}
                if header != '':
                    row_data[header]['value'] = cell.get("text")
                    links = cell.get('links')
                    for link in links:
                        if link.get('wiki_title'):
                            row_data[header]['wiki_title'] = link['wiki_title']
                        if link.get('url'):
                            row_data[header]['url'] = link['url']
            columns_info.append(row_data)
        return columns_info

    def extract_rows(self, table_information):
        """
        提取有效的列名，即不为空字符串的列
        """
        headers = [col.get("column_name") for col in table_information['columns']]
        result_rows = [
            [cell.get('text').upper() for header, cell in zip(headers, row) if header != ""]
            for row in table_information['rows']
        ]
        return result_rows
    
    def get_table_wikititles(self, table_information):
        wiki_titles = []
        table_rows = table_information['rows']
        for row in table_rows:
            for item in row:
                if 'wiki_title' in str(item):
                    for link in item['links']:
                        wiki_titles.append(link['wiki_title'])
        return wiki_titles

    def tabale2db(self, conn, id_mapping, table_id, table_row_count = None, postprocess_flag = 0):
        cursor = conn.cursor()
        table_information = self.get_table_by_id(id_mapping, table_id)
        table_name = f'table_{table_id}'
        rows = extract_rows(table_information)
        insert_error = 0#UNIQUE constraint failed
        if postprocess_flag:
            if table_row_count != len(rows):
                insert_error = 1
                print(table_id)
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                

        column_data, key_column = self.get_table_columns(table_information)
        
        column_definitions = []
        pos = 0
        for column_name, column_type in column_data: 
            if column_type == 'Unknown':
                column_type = 'text'
            if column_name != '':
                pos+=1
                if insert_error == 1 and postprocess_flag:
                    column_definitions.append(f'col{pos}_{table_id} {column_type}')
                if postprocess_flag == 0:
                    if (column_name, pos-1) in key_column:
                        column_definitions.append(f'col{pos}_{table_id} {column_type} PRIMARY KEY')
                    else:
                        column_definitions.append(f'col{pos}_{table_id} {column_type}')
        
        column_definitions_str = ", ".join(column_definitions)
        if not column_definitions:
            cursor.close()
            return 

        create_table_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_definitions_str});'
        cursor.execute(create_table_query)
        conn.commit()
        
        placeholders = ", ".join(["?"] * pos)  # 占位符
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders});"
        cursor.executemany(insert_query, rows)
        conn.commit()
        cursor.close()

    def tables2db(self, db_path=None):
        db_path = db_path or self.db_path
        table_id_list = list(set(self.data_processor.get_question_table_id()))
        print(len(table_id_list))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        table_rows = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_rows[table_name] = count

        cursor.close()

        table_name_list = [table[0].split("table_")[1] for table in tables]
        # table_id_list = list(set(table_id_list) - set(table_name_list)) #在处理所有表的过程中中断
        id_mapping = self.get_table_mapping()
        for table_id in table_id_list:
            try:
                # print(f'正在处理 {table_id}')
                # tabale2db(conn, id_mapping, table_id) 
                self.tabale2db(conn, id_mapping, table_id, table_rows[f'table_{table_id}'], 1)#后处理
            
            except Exception as e:
                print(f"出错了: {table_id} {e}")

        conn.close()
    


class TableKGProcessor:
    def __init__(self):
        self.kg_path = '/home/cmf/multiQA/KG_LLM/kg.db'
        self.table_processor = TableProcessor()
        self.data_processor = MMQADataProcessor()

    def table_as_entities(self, table_information):
        """
        将表格转换为实体形式
        """
        columns_info = self.table_processor.extract_column_values_by_id(table_information)
        entities = []
        for row in columns_info:
            entity = {}
            for column_name, cell_value in row.items():
                entity[column_name] = cell_value
            entities.append(entity)
        return entities

    def initialize_table_entity(self, entities, key, description, title, table_name, columns, is_column=0, is_column_value=0, wiki_title='', url ='', column=''):
        key = key.upper()
        entities[key] = {
            'description': description.upper(),
            'title': title.upper(),
            'table_name': table_name.upper(),
            'is_column': is_column,
            'is_column_value': is_column_value,
            'wiki_title': wiki_title.upper(),
            'url': url,
            'column': column.upper(),
            'columns': columns.upper()
        }

    def get_table_kg_data(self, id_mapping, table_id, entities):
        table_information = self.table_processor.get_table_by_id(id_mapping, table_id)
        column_rows = self.table_processor.extract_column_rows(table_information)
        title = table_information['title']
        name = table_information['name']
        table_name = f'{title}_{name}'.replace(' ','_')
        column_data, key_column = self.table_processor.get_table_columns(table_information)
        column_name_list = []
        for column_name, column_type in column_data:    
            if column_name != '':
                # column_name = normalized_string(column_name)
                column_name_list.append(column_name.replace(' ', '_'))
        
        # for column_name, column_type in column_data:
        #     # column_name = normalized_string(column_name)
        #     description = f'{column_name} is about {title} of {name}"'
        #     self.initialize_table_entity(entities, column_name.replace(' ', '_'), description, title, table_name, '@#'.join(column_name_list),1)
        
        for column_row in column_rows:
            # print(column_row)
            row_list = []
            for key, value in column_row.items():
                row_value = value.get('value')
                row_list.append(row_value)
            
            for key, value in column_row.items():
                
                row_value = value.get('value')
                wiki_title = value.get('wiki_title') if value.get('wiki_title') else ''
                url = value.get('url') if value.get('url') else ''
                # print('----------', row_value)
                # print(alias)
                columns = '@#'.join(column_name_list)
                row_data = '@#'.join(str(item) if item is not None else '' for item in row_list)
                
                description = f'This is about "{name}" of "{title}" and contains informations of {row_value}, contains the following information: {columns} (informations are separated by "@#"). The details are {row_data}'
                if row_value and row_value!='' and row_value!='-' and row_value!='_' and row_value!='—':
                    self.initialize_table_entity(entities, row_value, description, title, table_name, '@#'.join(column_name_list), 0, 1, wiki_title, url, key)

        description = f'This is about "{name}" of "{title}",which contains detailed informationis.'
        self.initialize_table_entity(entities, name, description, title, table_name, '@#'.join(column_name_list))
        description = f'This is about "{name}" of "{title}"'
        self.initialize_table_entity(entities, title, description, title, table_name, '@#'.join(column_name_list))
        
        return entities

    def get_tables_kg_data(self):
        entities = {}
        id_mapping = self.table_processor.get_table_mapping()
        table_id_list = list(set(self.data_processor.get_question_table_id()))
        for table_id in table_id_list:
            entities = self.get_table_kg_data(id_mapping, table_id, entities)

    def get_cols_by_table_id(self, table_id, kg_path=None):
        kg_path = kg_path or self.kg_path
        conn = sqlite3.connect(kg_path)
        cursor = conn.cursor()
        sql = 'select columns from entities where data_id = ?'
        cursor.execute(sql, (table_id,))
        columns = cursor.fetchone()[0]
        columns_list = columns.split('@#')
        return columns_list

    def get_tableId_by_name(self, table_name, kg_path=None):
        '''
        根据sql中的table name映射到id，以便后续得到真正的tablename（table_<table_id>）和col(col<id>_<table_id>)
        '''
        kg_path = kg_path or self.kg_path
        conn = sqlite3.connect(kg_path)
        cursor = conn.cursor()
        sql = 'select table_value from table_hash where table_key = ?'
        cursor.execute(sql, (table_name,))
        table_row = cursor.fetchone()

        if table_row is not None:
            table_id = table_row[0]
        else:
            table_id = None  

        cursor.close()
        conn.close()
        return table_id
    
    def get_tableCol_by_col(self, col, table_id, kg_path=None):
        kg_path = kg_path or self.kg_path
        conn = sqlite3.connect(kg_path)
        cursor = conn.cursor()
        sql = 'SELECT table_key FROM table_hash WHERE table_value = ? AND table_key LIKE ?'
        cursor.execute(sql, (col, f'%{table_id}%'))
        table_col_name = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return table_col_name

    def create_table_hash(self, source_kg_path, target_kg_path):
        source_db = sqlite3.connect(source_kg_path) 
        target_db = sqlite3.connect(target_kg_path) 

        source_cursor = source_db.cursor()
        target_cursor = target_db.cursor()

        target_cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_hash (
            table_key TEXT PRIMARY KEY,
            table_value TEXT
        )
        ''')

        source_cursor.execute('SELECT table_key, table_value FROM table_hash')

        rows = source_cursor.fetchall()  
        for row in rows:
            target_cursor.execute('''
            INSERT OR REPLACE INTO table_hash (table_key, table_value)
            VALUES (?, ?)
            ''', (row[0], row[1]))

        target_db.commit()

        source_cursor.close()
        target_cursor.close()
        source_db.close()
        target_db.close()

        print("Data migration completed successfully.")

            
    def tablehash(self, kg_path=None):
        kg_path = kg_path or self.kg_path
        conn = sqlite3.connect(kg_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS table_hash")
        conn.commit()
        cursor.close()

        conn.execute('''
        CREATE TABLE IF NOT EXISTS table_hash (
            table_key TEXT PRIMARY KEY,
            table_value TEXT
        )
        ''')
        conn.commit()

        table_hash = {}
        id_mapping = TableProcessor().get_table_mapping()  # Get table mapping from MMQADataProcessor
        table_id_list = list(set(self.data_processor.get_question_table_id()))  # Get table ids
        insert_query = "INSERT OR IGNORE INTO table_hash (table_key, table_value) VALUES (?, ?)"
        for table_id in table_id_list:
            cursor = conn.cursor()
            result = []
            table_information = id_mapping.get(table_id, {})
            table_name = f"{table_information['title']}_{table_information['name']}"
            table_name = table_name.replace(' ', '_')
            table_hash[f'table_{table_id}'] = table_name
            table_hash[table_name] = f'table_{table_id}'
            result.append((f'table_{table_id}', table_name.upper()))
            result.append((table_name.upper(), table_id))
            column_data, key_column = TableProcessor.get_table_columns(table_information)
            pos = 0
            for column_name, column_type in column_data:
                if column_name != '':
                    pos += 1
                    column = f'col{pos}_{table_id}'
                    column_name = column_name.replace(' ', '_')
                    table_hash[column] = column_name
                    result.append((column, column_name.upper()))
            cursor.executemany(insert_query, result)
            conn.commit()
            cursor.close()

        conn.close()
        return table_hash


class MMQAProcessor:
    def __init__(self):
        self.data_processor = MMQADataProcessor()
        self.table_processor = TableProcessor()
        self.kg_processor = KGProcessor()
    
    def process(self):
        # Example of integrating the processes
        self.kg_processor.tablehash()
        # Add other processing logic if needed

class KGConstructor:
    def __init__(self):
        self.data_processor = MMQADataProcessor()
        self.table_processor = TableProcessor()
        self.table_kg_processor = TableKGProcessor()
        self.image_processor = ImageProcessor()
        self.text_db_path = '/data/cmf/mmqa/kg/kg_llama3-8b.db'
        self.kg_path = '/data/cmf/mmqa/kg/kg_new.db'
    
    def text_kg(self):
        text_mapping = self.data_processor.create_text_mapping()
        db_path = self.text_db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = 'SELECT data_id, entity, entity_type, type, title, description, text from entities'
        cursor.execute(sql)
        entity_results = cursor.fetchall()
        uppercase_results = []
        data_id_list = []
        for data_id, entity, entity_type, type_, title, description, text in entity_results:
            data_id_list.append(data_id)
        data_id_list = set(list(data_id_list))
        # for data_id in data_id_list:
        #     entity = text_mapping.get(data_id, {}).get('title', '')
        #     text = text_mapping[data_id]['text']
        #     url = text_mapping.get(data_id, {}).get('url', '')
        #     uppercase_results.append((data_id, entity.upper(), '', 'text', entity.upper(), text.upper(), text.upper(), url))
        for data_id, entity, entity_type, type_, title, description, text in entity_results:
            url = text_mapping.get(data_id, {}).get('url', '')
            uppercase_results.append((
                data_id, 
                safe_upper(entity), 
                safe_upper(entity_type), 
                type_,
                safe_upper(title), 
                safe_upper(description), 
                safe_upper(text), 
                url
            ))

        sql = 'SELECT data_id, head_entity, tail_entity, relation, description, text, type, strength from relationships'
        cursor.execute(sql)
        relation_results = cursor.fetchall()
        cursor.close()
        conn.close()
        uppercase_relation_results = [
            (data_id, 
            safe_upper(head_entity), 
            safe_upper(tail_entity), 
            safe_upper(relation), 
            safe_upper(description), 
            safe_upper(text), 
            type_,
            strength )
            for data_id, head_entity, tail_entity, relation, description, text, type_, strength in relation_results
        ]

        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        entity_insert_query = '''
        INSERT INTO entities 
        (data_id, entity, entity_type, type, title, description, text, url) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(entity_insert_query, uppercase_results)
        conn.commit()

        relation_insert_query = '''
        INSERT INTO relationships 
        (data_id, head_entity, tail_entity, relation, description, text, type, strength) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(relation_insert_query, uppercase_relation_results)
        conn.commit()
        cursor.close()
        conn.close()

    def generate_table_relations_with_equivalences(self, table_data):
        """
        根据表格数据生成完整的知识图谱三元组 (head_entity, relation, tail_entity)，并处理等价实体。
        :param table_data: 包含表格数据的字典。
        :return: 知识图谱的三元组列表。
        """
        knowledge_graph = []
        equivalences = []  # 存储等价实体的三元组

        rows = table_data["table"]["table_rows"]
        columns = table_data["table"]["header"]
        table_name = table_data["table"].get("table_name", "")
        title = table_data.get("title", "")
        table_id = table_data['id']
        if title:
            knowledge_graph.append((table_id, table_name, "belongs to", title, 'table'))

        # 提取列名
        column_names = [col["column_name"] for col in columns]
        key_pos, null_cols = [], []
        for i in range(0, len(columns)):
            col = columns[i]
            if col['metadata'].get('is_key_column'):
                key_pos.append(i)
            if not col['column_name'] or col['column_name'] == '':
                null_cols.append(i)
            else:
                if table_name:
                    knowledge_graph.append((table_id, col['column_name'], "the table name is", table_name, 'table'))

        # 遍历表格的每一行
        alias = {}
        for row in rows:
            head_entity = None
            for i, cell in enumerate(row):
                if i in null_cols:
                    continue
                head_entity = ''
                # 处理头实体（假定 "Team" 列为头实体列）
                if i in key_pos:  # 假定第二列 "Team" 是头实体
                    head_entity = cell['text']
                    # knowledge_graph.append((table_id, head_entity, "is", column_names[i], 'table'))
                    links = cell.get('links')
                    for link in links:
                        if link.get('wiki_title'):
                            if cell['text'] != link.get('wiki_title'):
                                alias[link.get('wiki_title')] = cell["text"]
                                alias[cell["text"]] = link.get('wiki_title')
                                knowledge_graph.append((table_id, cell["text"], "equivalent to", link.get('wiki_title'), 'table'))
                                # knowledge_graph.append((table_id, link.get('wiki_title'), "equivalent_to", cell['text'], 'table'))

                if head_entity:
                    for j, cell in enumerate(row):
                        if j != i and j not in null_cols:
                            relation = f'the {column_names[j]} of this {column_names[i]} is'
                            tail_entity = cell['text']  
                            knowledge_graph.append((table_id, head_entity, relation, tail_entity, 'table'))

                            links = cell.get('links')
                            for link in links:
                                if link.get('wiki_title'):
                                    if cell['text'] != link.get('wiki_title'):
                                        alias[link.get('wiki_title')] = cell["text"]
                                        alias[cell["text"]] = link.get('wiki_title')
                                        knowledge_graph.append((table_id, cell["text"], "equivalent to", link.get('wiki_title'), 'table'))
                                        # knowledge_graph.append((table_id, link.get('wiki_title'), "equivalent_to", cell['text'], 'table'))

        return knowledge_graph

    def table_kg(self):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()

        table_id_list = list(set(self.data_processor.get_question_table_id()))
        insert_query = '''
        INSERT INTO entities 
        (data_id, entity, type, title, description, isColumn, isColumnValue, table_name, columns, column, wiki_title, url) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        id_mapping = self.table_processor.get_table_mapping()
        for table_id in table_id_list:
            entities = {}
            result = []
            entities = self.table_kg_processor.get_table_kg_data(id_mapping, table_id, entities)
            for entity_name, entity_values in entities.items():
                title = entity_values['title']
                table_name = entity_values['table_name']
                is_column = entity_values['is_column']
                is_column_value = entity_values['is_column_value']
                wiki_title = entity_values['wiki_title']
                url = entity_values['url']
                column = entity_values['column']  
                columns = entity_values['columns']
                description = entity_values['description']
                result.append((table_id, entity_name, 'table', title, description, is_column, is_column_value, table_name, columns, column, wiki_title, url))
            cursor.executemany(insert_query, result)
            conn.commit()


            relations_list = self.generate_table_relations_with_equivalences(id_mapping[table_id])
            relation_insert_query = '''
            INSERT INTO relationships 
            (data_id, head_entity, relation, tail_entity, type) 
            VALUES (?, ?, ?, ?, ?)
        '''
            cursor.executemany(relation_insert_query, relations_list)
            conn.commit()


        cursor.close()
        conn.close()


    def initialize_image_entity(self, entities, key, title, description, text, isImageTitle, url, path):
        key = key.upper()
        entities[key] = {
            'title': title.upper(),
            'description': description.upper(),
            'text': text.upper(),
            'isImageTitle': isImageTitle,
            'url':url,
            'path':path
        }
    

    def image_kg(self, entity_conf_value=0.2, attribute_conf_value=0.2):
        questions, image_q, table_q, image_q = self.data_processor.get_dev_information()
        image_mapping = self.image_processor.create_image_mapping()
        image_list = []
        image_data_list = []
        entity_result = []
        for question in questions:
            image_list += question["image_doc_ids"]
        image_list = list(set(image_list))

        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        entity_insert_query = '''
        INSERT INTO entities 
        (data_id, entity, type, title, description, text, isImageTitle, image_path, url) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        relation_insert_query = '''
        INSERT INTO relationships 
        (data_id, head_entity, tail_entity, relation, description, text, type) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''

        for image_id in image_list:
            entity_result = []
            image_data = image_mapping[image_id]
            title = image_data['title']
            url = image_data['url']
            result_dict = image_data['object']
            image_path = image_data['path']
            caption = image_data['caption']

            entities = {}
            relations = []

            #在manymodalqa和webqa中，需要看一下title在不在实体表中，不在才加
            self.initialize_image_entity(entities, title, title,  f'{title} describes that {caption}', f'{caption}', 1, url, image_path)
            for entity in result_dict:
                attributes_str = ','.join(list(set(result_dict[entity])))
                attribute_entity = f'{attributes_str} {entity}'
                self.initialize_image_entity(entities, entity, title, f'{title} has {attribute_entity}', f'{caption}', 0, url, image_path)
                relations.append((image_id, title.upper(), entity.upper(), 'HAS', f'{title.upper()} HAS {attribute_entity.upper()}', f'{caption.upper()}', 'image'))
            
            for entity_name, entity_values in entities.items():
                title = entity_values['title']
                description = entity_values['description']
                text = entity_values['text']
                isImageTitle = entity_values['isImageTitle']
                url = entity_values['url']
                entity_result.append((image_id, entity_name, 'image', title, description, text, isImageTitle, image_path, url))

            cursor.executemany(entity_insert_query, entity_result)
            conn.commit()
            cursor.executemany(relation_insert_query, relations)
            conn.commit()
        cursor.close()
        conn.close()    

    def create_KG_table_entity_link(self):
        questions,text_q, table_q, image_q = self.data_processor.get_dev_information()
        KG = KnowledgeGraph(self.kg_path)
        KG.drop_entity_links()
        # for idx, question_data in enumerate(questions):
        print(len(questions))
        for idx in range(0,len(questions)):
            data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
            kg = KG.create_kg_view(data_id_list)
            # KG.create_entity_link(idx)
            kg = KG.create_entity_text_link(idx)
    
    def transfer_all_entity_links(self, src_db, dest_db):
        """
        将 src_db 中 entity_links 表中 type 和 link_type 为 'text' 的记录插入到 dest_db 中的 entity_links_new 表。
        
        :param src_db: 源数据库的路径
        :param dest_db: 目标数据库的路径
        """
        # 连接到源数据库和目标数据库
        conn_src = sqlite3.connect(src_db)
        conn_dest = sqlite3.connect(dest_db)
        
        # 创建游标对象
        cursor_src = conn_src.cursor()
        cursor_dest = conn_dest.cursor()

        try:
            sql = 'DROP TABLE IF EXISTS entity_links'       
            cursor_dest.execute(sql)
            cursor_dest.execute('''
                CREATE TABLE IF NOT EXISTS entity_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_id TEXT,
                    entity_name TEXT,
                    link_data_id TEXT,
                    link_entity_name TEXT,
                    type TEXT,
                    link_type TEXT,
                    kg_id INTEGER
                );
                ''')
            conn_dest.commit()  # 提交表的创建
            print("Table entity_links created or already exists")

            # 查询语句：从源数据库中获取 type 和 link_type 为 'text' 的记录
            select_sql = """
                SELECT data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id
                FROM entity_links_new;
            """
            
            # 执行查询
            cursor_src.execute(select_sql)

            # 获取所有符合条件的记录
            rows = cursor_src.fetchall()

            if rows:
                # 插入到目标数据库的 entity_links_new 表中
                insert_sql = """
                    INSERT INTO entity_links (data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """

                # 批量插入记录
                cursor_dest.executemany(insert_sql, rows)

                # 提交事务
                conn_dest.commit()
                print(f"{len(rows)} records inserted successfully.")
            else:
                print("No matching records found to insert.")

        except Exception as e:
            print(f"Error during database operations: {e}")
        finally:
            # 关闭游标和数据库连接
            cursor_src.close()
            cursor_dest.close()
            conn_src.close()
            conn_dest.close()

    
    def transfer_text_entity_links(self, src_db, dest_db):
        """
        将 src_db 中 entity_links 表中 type 和 link_type 为 'text' 的记录插入到 dest_db 中的 entity_links_new 表。
        
        :param src_db: 源数据库的路径
        :param dest_db: 目标数据库的路径
        """
        # 连接到源数据库和目标数据库
        conn_src = sqlite3.connect(src_db)
        conn_dest = sqlite3.connect(dest_db)
        
        # 创建游标对象
        cursor_src = conn_src.cursor()
        cursor_dest = conn_dest.cursor()

        try:
            # 查询语句：从源数据库中获取 type 和 link_type 为 'text' 的记录
            select_sql = """
                SELECT data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id
                FROM entity_links_new_1
                WHERE (type = 'text' and link_type = 'text') or (type = 'text' and link_type = 'image') or  (type = 'image' and link_type = 'text');
            """
            
            # 执行查询
            cursor_src.execute(select_sql)

            # 获取所有符合条件的记录
            rows = cursor_src.fetchall()

            if rows:
                # 插入到目标数据库的 entity_links_new 表中
                insert_sql = """
                    INSERT INTO entity_links_new (data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """

                # 批量插入记录
                cursor_dest.executemany(insert_sql, rows)

                # 提交事务
                conn_dest.commit()
                print(f"{len(rows)} records inserted successfully.")
            else:
                print("No matching records found to insert.")

        except sqlite3.Error as e:
            print(f"Error during database operations: {e}")
        finally:
            # 关闭游标和数据库连接
            cursor_src.close()
            cursor_dest.close()
            conn_src.close()
            conn_dest.close()
    
    def delete_entity_link(self):
        KG = KnowledgeGraph(self.kg_path) 
        KG.remove_entity_link_duplicates()
        KG.delete_entity_link()
    

    def create_kg_table(self):
        print(self.kg_path)
        conn = sqlite3.connect(self.kg_path)
        # Create the table if it does not exist
        conn.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id TEXT,
            entity TEXT,
            entity_type TEXT,
            type TEXT,
            title TEXT,
            description TEXT,
            text TEXT,
            isColumn INTEGER DEFAULT 0, 
            isColumnValue INTEGER DEFAULT 0, 
            table_name TEXT, 
            columns TEXT, 
            column TEXT, 
            isImageTitle INTEGER DEFAULT 0,
            wiki_title TEXT,
            url TEXT, 
            image_path TEXT
        );
        ''')
        conn.commit()
        conn.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id TEXT,
            head_entity TEXT,
            tail_entity TEXT,
            relation TEXT,
            description TEXT,
            text TEXT,
            type TEXT,
            strength INT
        );
        ''')
        conn.commit()
        conn.close()

    def main(self):
        # self.create_kg_table()
        # self.text_kg()
        # self.image_kg()
        # self.table_kg()
        # self.create_KG_table_entity_link()
        src_db = '/data/cmf/mmqa/kg/kg_new.db'
        dest_db = '/data/cmf/mmqa/kg/kg.db'
        # self.delete_entity_link()
        # self.transfer_text_entity_links(src_db, dest_db)
        self.transfer_all_entity_links(src_db, dest_db)
        

# mmqa_kg = KGConstructor()
# mmqa_kg.main()

# processor = TableKGProcessor()
# source_kg_path = '/home/cmf/multiQA/KG_LLM/kg.db'
# target_kg_path = '/data/cmf/mmqa/kg/kg_new.db'
# processor.create_table_hash(source_kg_path, target_kg_path)
