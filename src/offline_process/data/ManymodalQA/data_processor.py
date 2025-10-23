import json
import io
import re
import csv
import pandas as pd
import os
import sqlite3
from collections.abc import Iterator
from itertools import islice
import tiktoken
from config.config import config_loader
from utils.utils import read_jsonl_file
from KG_LLM.offline_process.kg import KnowledgeGraph

def safe_upper(value):
    return value.upper() if value else value
    
class ManymodalQADataProcessor:
    def __init__(self):
        self.test_path = config_loader.get_dev_path('ManymodalQA')
        self.question_path = '/data/cmf/manymodalqa/Manymodal_dev_question_split.jsonl'
    def load_data(self):
        with open(self.test_path, 'r') as file:
            data = json.load(file)
        return data

    def extract_question_information(self, data):
        extracted_info = {}
        extracted_info["question"] = data.get("question", "")
        extracted_info["qid"] = data.get("id", "")
        extracted_info["answer"] = data.get('answer')
        extracted_info["image"] = data.get('image')
        
        extracted_info["text"] = data.get('text')
        extracted_info['table'] = data.get('table')
        extracted_info['type'] = data.get('q_type')
        extracted_info['modalities'] = data.get('q_type')
    
        return extracted_info
    def get_questions_information(self):
        data_list = read_jsonl_file(self.question_path)
        question_data_list = []
        for data in data_list:
            extracted_info = {}
            extracted_info["question"] = data.get("question", "")
            extracted_info["qid"] = data.get("qid", "")
            extracted_info["answer"] = data.get('answer')
            extracted_info["image_doc_ids"] = data.get('image', [])
            
            extracted_info["text_doc_ids"] = data.get('text')
            extracted_info['table_id'] = data.get('table', [])
            extracted_info['type'] = data.get('type')
            extracted_info['modalities'] = data.get('modalities')
            question_data_list.append(extracted_info)
        return question_data_list
    def get_dev_information(self, test_path = None):
        test_path = test_path or self.test_path

        with open(test_path, 'r') as file:
            data_list = json.load(file)
        result = []
        question_data_list = []
        for data in data_list:  
            extracted_info = self.extract_question_information(data)
            question_data_list.append(extracted_info)

        return question_data_list

class TextProcessor:
    def __init__(self, max_tokens):
        self.data_processor = ManymodalQADataProcessor()
        self.max_tokens_per_chunk = max_tokens
    
    def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
        """Return the number of tokens in the given text."""
        if token_encoder is None:
            token_encoder = tiktoken.get_encoding("cl100k_base")
        return len(token_encoder.encode(text))  # type: ignore

    def batched(self, iterable: Iterator, n: int):
        """
        Batch data into tuples of length n. The last batch may be shorter.

        Taken from Python's cookbook: https://docs.python.org/3/library/itertools.html#itertools.batched
        """
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            value_error = "n must be at least one"
            raise ValueError(value_error)
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    def chunk_text(self, 
        text: str, max_tokens: int, token_encoder: tiktoken.Encoding | None = None
    ):
        """Chunk text by token length, ensuring that sentences are not split."""
        if token_encoder is None:
            token_encoder = tiktoken.get_encoding("cl100k_base")
        tokens = token_encoder.encode(text)  # type: ignore
        chunks = []
        current_chunk = []
        current_length = 0

        # Split text into sentences to avoid splitting mid-sentence
        sentences = re.split(r'(?<=[。！？.!?])', text)
        
        for sentence in sentences:
            sentence_tokens = token_encoder.encode(sentence)
            if current_length + len(sentence_tokens) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence_tokens
                current_length = len(sentence_tokens)
            else:
                current_chunk.extend(sentence_tokens)
                current_length += len(sentence_tokens)

        if current_chunk:
            chunks.append(current_chunk)

        yield from (token_encoder.decode(chunk) for chunk in chunks)

    def create_text_mapping(self):
        question_datas_list = self.data_processor.get_dev_information()
        data_list = []
        data_id_list = []
        data = {}
        for question_data in question_datas_list:
            data_id = question_data['id']
            text = question_data['text']
            chunks = list(self.chunk_text(text, self.max_tokens_per_chunk))
            for idx, chunk in enumerate(chunks):
                chunk_id = f'{data_id}_{idx}'
                data_list.append({'id':chunk_id, 'text':chunk})
                data_id_list.append(chunk_id)
                data[chunk_id] ={}
                data[chunk_id] = {'id':chunk_id, 'text':chunk}
        return data
        
    def extract_text_id_list(self):
        file_path = '/data/cmf/manymodalqa/Manymodal_dev_question_split.jsonl'
        question_datas_list =  self.data_processor.get_dev_information()
        data_list = []
        data_id_list = []
        data = {}
        question_data_dict_list = []
        for question_data in question_datas_list:
            question_data_dict = {}
            data_id = question_data['qid']
            question_data_dict['qid'] = data_id
            question_data_dict['question'] = question_data['question']
            question_data_dict['type'] = question_data['type']
            question_data_dict['modalities'] = question_data['modalities']

            if question_data['table']:
                question_data_dict['table'] = [data_id]

            if question_data['image']:
                question_data_dict['image'] = [data_id]
                data_list.append({'id':data_id, 'image':question_data['image']})
                data_id_list.append(data_id)
                data[data_id] = {'id':data_id, 'image':question_data['image']}

            text_id_list = []
            text = question_data['text']
            chunks = list(self.chunk_text(text, self.max_tokens_per_chunk))
            for idx, chunk in enumerate(chunks):
                chunk_id = f'{data_id}_{idx}'
                text_id_list.append(chunk_id)
                data_list.append({'id':chunk_id, 'text':chunk})
                data_id_list.append(chunk_id)
                data[chunk_id] ={}
                data[chunk_id] = {'id':chunk_id, 'text':chunk}
            question_data_dict['text'] = text_id_list
            with open(file_path, "a", encoding="utf-8") as jsonl_file:
                jsonl_file.write(json.dumps(question_data_dict, ensure_ascii=False) + "\n")
                # print(f'写入{data_id}成功')
            
        all_text_id = list(set(data_id_list))
        return all_text_id

        # conn = sqlite3.connect(f'{args.dataset}_kg_{args.engine}.db')
        # cursor = conn.cursor()
        # cursor.execute('select distinct(data_id) from entities')
        # used_data_list = [row[0] for row in cursor.fetchall()]
        # data_id_list = list(set(data_id_list) - set(used_data_list))
        # cursor.close()
        # conn.close()
        # for d_id in data_id_list:
        #     text_data = data[d_id]
        #     data_list.append(text_data)

class ImageProcessor:
    def __init__(self):
        self.image_path = '/data/cmf_dataset/ManymodalQA/dev_images'
        self.data_processor = ManymodalQADataProcessor()
        self.imgae_caption_path = '/data/cmf/manymodalqa/ManyModalQA_dev_image_facts_with_caption.jsonl'
        self.object_path = '/data/cmf/manymodalqa/image_object'
    def create_image_mapping(self):
        data_list = read_jsonl_file(self.imgae_caption_path)
        id_mapping = {}
        for data in data_list:
            image_id = data.get('id')
            title = data.get('title')
            path = data.get('path')
            url = data.get('url')
            caption = data.get('caption')
            if image_id:
                object_dict = self.get_image_object(image_id, 0.2, 0.2)
                id_mapping[image_id] = {'id': image_id, 'title': title, 'url': url, 'path': self.image_path+'/'+path , 'type': 'image', 'caption': caption, 'object':object_dict}
        return id_mapping

    def get_image_object(self, image_id, entity_conf_value, attribute_conf_value):
        file_path = os.path.join(self.object_path, f'{image_id}.txt' )
        if not os.path.exists(file_path):
            return {}
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
        
        self.db_path = '/data/cmf/manymodalqa/ManymodalQA.db'
        self.table_file_path = '/data/cmf/manymodalqa/ManyModalQA_dev_table_facts.jsonl'
    
    def create_table_mapping(self):
        data_list = read_jsonl_file(self.table_file_path)
        id_mapping = {}
        for data in data_list:
            column_list = data['header']
            # columns = '@#'.join(column_list)
            data_id = data['id']
            rows_list = data['rows']
            id_mapping[data_id] = {'id': data_id, 'columns': column_list, 'rows': rows_list}
        return id_mapping

    def create_table_from_data(self, data_list):
        """根据每个数据字典的 table 字段创建表。"""
        conn = sqlite3.connect(self.db_path)  
        cursor = conn.cursor()

        error = 0

        for data in data_list:
            try:
                if 'id' in data and 'table' in data and data['table'] is not None:
                    table_name = data['id']  # 使用第一个 id 作为表名
                    table_data = data['table']
                    lines = table_data.strip().split('\n')
                    
                    rows = []
                    for line in lines:
                        reader = csv.reader(io.StringIO(line))
                        for row in reader:
                            rows.append(row)
                            
                    if len(rows) < 2:
                        print(f"Table for {table_name} does not have enough data.")
                        continue
                    
                    original_columns = [col.strip() for col in rows[0]]
                    column_counts = {}
                    unique_columns = []

                    for col in original_columns:
                        if col in column_counts:
                            column_counts[col] += 1
                            unique_columns.append(f"{col}_{column_counts[col]}")  # 重命名为 col{i}
                        else:
                            column_counts[col] = 1
                            unique_columns.append(col)

                    columns_sql = ', '.join([f'"{col}" TEXT' for col in unique_columns])  # 使用 TEXT 类型
                    
                    # 创建表
                    cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql});')

                    for idx, row in enumerate(rows[1:]):
                        try:
                            values = [val.strip().replace('"', '""') for val in row]
                            placeholders = ', '.join(['?'] * len(values))
                            cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', values)
                            conn.commit()
                        except Exception as e:
                            pattern = r'"([^"]*(?:""[^"]*)*)"'
                            row = re.findall(pattern, lines[idx+1])
                            values = [val.strip().replace('"', '""') for val in row]
                            placeholders = ', '.join(['?'] * len(values))
                            cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', values)
                            
                    conn.commit() 

                else:
                    d=0
            
            except Exception as e:
                error += 1
                print(f"Exception occurred for table '{table_name}': {e}")
        
        conn.commit()
        conn.close()
        print(error)
    
    def csv_to_sqlite(folder_path, db_path):
        # 连接到 SQLite 数据库（如果不存在会自动创建）
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 遍历文件夹中的所有 CSV 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # 确保只处理 CSV 文件
                # 去掉文件扩展名，作为表名
                table_name = os.path.splitext(file_name)[0]
                


                # 确保表名符合 SQLite 命名规则（去掉非法字符）
                table_name = ''.join(c for c in table_name if c.isalnum() or c == '_')

                # 构造 CSV 文件的完整路径
                csv_file_path = os.path.join(folder_path, file_name)

                # 使用 pandas 读取 CSV 文件
                df = pd.read_csv(csv_file_path)

                # 将 DataFrame 写入 SQLite 数据库
                df.to_sql(table_name, conn, if_exists='replace', index=False)

                print(f"Inserted data from {file_name} into table {table_name}")

        # 关闭数据库连接
        conn.close()

    def count_non_empty_cells(self, cursor, table_name):
        """
        计算指定表中所有列的非空单元格总数。
        """
        # 获取表的所有列名
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = [row[1] for row in cursor.fetchall()]

        # 初始化非空单元格计数
        total_non_empty_cells = 0

        # 遍历每一列，统计非空单元格数量
        for column in columns:
            cursor.execute(f"SELECT COUNT(`{column}`) FROM `{table_name}` WHERE `{column}` IS NOT NULL AND `{column}` != '';")
            count = cursor.fetchone()[0]
            total_non_empty_cells += count

        return total_non_empty_cells

    def compare_records(self, db1_path, db2_path):
        """
        比较两个数据库中表的非空单元格总数和记录数。
        """
        # 连接到两个数据库
        conn1 = sqlite3.connect(db1_path)
        conn2 = sqlite3.connect(db2_path)
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

        # 获取第一个数据库的所有表名
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor1.fetchall()

        # 遍历每个表
        for table in tables:
            table_name = table[0]

            # 计算 db1 中该表的非空单元格总数
            non_empty_cells_db1 = count_non_empty_cells(cursor1, table_name)

            # 在第二个数据库的 entities 表中查找 data_id 为该表名的记录数
            cursor2.execute("SELECT COUNT(*) FROM entities WHERE data_id = ? and entity!='-' and entity!=''", (table_name,))
            count_db2 = cursor2.fetchone()[0]

            # 比较非空单元格总数和记录数
            if non_empty_cells_db1 >= count_db2:
                flag = 1
                # print(f"Table '{table_name}': Non-empty cell count matches record count ({non_empty_cells_db1})")
            else:
                print(f"Table '{table_name}': Non-empty cell count does NOT match record count "
                    f"(DB1: {non_empty_cells_db1}, DB2: {count_db2})")

        # 关闭数据库连接
        conn1.close()
        conn2.close()

class KGConstructor:
    def __init__(self):
        self.question_processor = ManymodalQADataProcessor()
        self.text_processor = TextProcessor(max_tokens=600)
        self.image_processor = ImageProcessor()
        self.table_processor = TableProcessor()
        self.text_db_path = '/data/cmf/manymodalqa/kg/manymodalqa_kg_llama3-8b.db'
        self.kg_path = '/data/cmf/manymodalqa/kg/kg.db'
        self.table_kg_path = '/data/cmf/manymodalqa/kg/table.db'
    
    
    def text_kg(self):
        db_path = self.text_db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = "SELECT data_id, entity, entity_type, type, title, description, text from entities where entity!='' and entity != '∗' and entity!='**' and entity!='#' and entity!='$' and entity!='℃' and entity!='∗' and entity!='♭II' and entity!='₱'"
        cursor.execute(sql)
        entity_results = cursor.fetchall()
        text_uppercase_results = []
        image_uppercase_results = []

        text_mapping = self.text_processor.create_text_mapping()
        image_mapping = self.image_processor.create_image_mapping()
    
        for data_id, entity, entity_type, type_, title, description, text in entity_results:
            print(f"正在处理{data_id}, type:{type_}")
            if type_ == 'text':
                text_uppercase_results.append((
                data_id, 
                safe_upper(entity), 
                safe_upper(entity_type), 
                type_, 
                safe_upper(title),
                safe_upper(description), 
                safe_upper(text)
            ))
            else:
                url = image_mapping.get(data_id, {}).get('url', '')
                caption = image_mapping[data_id]['caption']
                
                image_path = image_mapping[data_id]['path']
                # image_path = os.path.join(self.image_processor.image_path, path)
                image_uppercase_results.append((
                    data_id, 
                    safe_upper(entity), 
                    safe_upper(entity_type), 
                    type_, 
                    safe_upper(description), 
                    safe_upper(text) + caption,
                    1, 
                    image_path,  
                    url  
                ))
                # print(entity)
                # print(text)
                # print(caption)
                image_uppercase_results.append((data_id, entity.upper(), entity_type.upper(), type_, description.upper(), text.upper() + caption,1, image_path, url))

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
            strength)
            for data_id, head_entity, tail_entity, relation, description, text, type_, strength in relation_results
        ]
       

        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        entity_insert_query = '''
        INSERT INTO entities 
        (data_id, entity, entity_type, type, title, description, text) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(entity_insert_query, text_uppercase_results)
        print(f'文本数据:{len(text_uppercase_results)} insert')
        conn.commit()

        entity_insert_query = '''
        INSERT INTO entities 
        (data_id, entity, entity_type, type, description, text, isImageTitle, image_path, url) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(entity_insert_query, image_uppercase_results)
        conn.commit()
        print(f'图像数据:{len(image_uppercase_results)} insert')

        relation_insert_query = '''
        INSERT INTO relationships 
        (data_id, head_entity, tail_entity, relation, description, text, type, strength) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(relation_insert_query, uppercase_relation_results)
        conn.commit()
        print(f'关系数据:{len(uppercase_relation_results)} insert')
        cursor.close()
        conn.close()

    def table_kg(self):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        result = []
        relations = []
        entities = {}

        # insert_query = '''
        # INSERT INTO entities 
        # (data_id, entity, type, description, isColumn, isColumnValue, table_name, columns, column) 
        # VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        # '''
        # id_mapping = self.table_processor.create_table_mapping()
        # print(f'一共有{len(id_mapping)}张表')

        # for table_id, value in id_mapping.items():
        #     result = []
        #     column_list = value['columns']
        #     rows_list = value['rows']
        #     columns = '@#'.join(column_list)

        #     for rows in rows_list:
        #         for idx, cell_value in enumerate(rows):
        #             if cell_value and cell_value != '' and cell_value != '-' and cell_value.lower()!='n/a' and cell_value.lower()!='none':
        #                 description = f'The {column_list[idx]} is {cell_value}'
        #                 result.append((table_id, cell_value.upper(), 'table', description.upper(), 0, 1, table_id, columns.upper(), column_list[idx].upper()))
            
        #     for column in column_list:
        #         description = f'It is a column of the table {table_id} .The columns in the table are {columns}(columns are separated by "@#")'
        #         result.append((table_id, column.upper(), 'table', description.upper(), 1, 0, table_id, columns.upper(), ''))
        #             # result.append((table_id, cell_value, 'table', ))
        #             # initialize_table_entity(entities, cell_value, description, table_id, columns, 1,  column=column_list[idx])
        #     # if len(result)>200:
        #     #     print(table_id)
        #     print(f'{len(result)} insert')
        #     cursor.executemany(insert_query, result)
        #     conn.commit()
        table_conn = sqlite3.connect(self.table_kg_path)
        table_cursor = table_conn.cursor()
        sql = 'SELECT data_id, head_entity, tail_entity, relation, description, type, strength from relationships'
        table_cursor.execute(sql)
        relation_results = table_cursor.fetchall()
        table_cursor.close()
        table_conn.close()

        uppercase_relation_results = [
            (data_id, 
            safe_upper(head_entity), 
            safe_upper(tail_entity), 
            safe_upper(relation), 
            safe_upper(description), 
            type_,
            strength)
            for data_id, head_entity, tail_entity, relation, description, type_, strength in relation_results
        ]

        relation_insert_query = '''
        INSERT INTO relationships 
        (data_id, head_entity, tail_entity, relation, description, type, strength) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
        cursor.executemany(relation_insert_query, uppercase_relation_results)
        conn.commit()
        print(f'关系数据:{len(uppercase_relation_results)} insert')

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

    def image_kg(self):
        

        image_mapping = self.image_processor.create_image_mapping()

        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        entity_insert_query = '''
        INSERT INTO entities 
        (data_id, entity, type, title, description, text, isImageTitle, image_path) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''

        relation_insert_query = '''
        INSERT INTO relationships 
        (data_id, head_entity, tail_entity, relation, description, text, type) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''

        for image_id, value in image_mapping.items():
            entity_result = []
            image_data = image_mapping[image_id]
            title = image_data['title']
            url = image_data['url']
            result_dict = image_data['object']
            image_path = image_data['path'] 
            caption = image_data['caption']
            entities = {}
            relations = []

            for entity in result_dict:
                attributes_str = ','.join(list(set(result_dict[entity])))
                attribute_entity = f'{attributes_str} {entity}'
                self.initialize_image_entity(entities, entity, title, f'{title} has {attribute_entity}', f'{caption}', 0, url, image_path)
                relations.append((image_id, title.upper(), entity.upper(), 'HAS', f'{title.upper()} HAS {attribute_entity.upper()}', f'{title.upper()} describes that {caption.upper()}', 'image'))

            for entity_name, entity_values in entities.items():
                title = entity_values['title']
                description = entity_values['description']
                text = entity_values['text']
                isImageTitle = entity_values['isImageTitle']
                entity_result.append((image_id, entity_name, 'image', title, description, text, isImageTitle, image_path))
           
            cursor.executemany(entity_insert_query, entity_result)
            conn.commit()
            cursor.executemany(relation_insert_query, relations)
            conn.commit()
        cursor.close()
        conn.close()    

    def postprocess(self):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        try:
            sql = "DELETE FROM entities WHERE entity IS NULL OR entity = '' OR entity = '-' or entity = '#';"
            cursor.execute(sql)
            conn.commit()

            sql = """
DELETE FROM relationships
WHERE 
    (head_entity IS NULL OR head_entity = '' OR head_entity = '*' OR head_entity = '**' OR head_entity = '#' OR head_entity = '$' OR head_entity = '℃' OR head_entity = '♭II' OR head_entity = '₱')
    OR
    (tail_entity IS NULL OR tail_entity = '' OR tail_entity = '*' OR tail_entity = '**' OR tail_entity = '#' OR tail_entity = '$' OR tail_entity = '℃' OR tail_entity = '♭II' OR tail_entity = '₱')
    OR
    (relation IS NULL OR relation = '' OR relation = '-')
    OR
    (description IS NULL OR description = '' OR description = '-');
"""
            cursor.execute(sql)
            conn.commit()

            query = """
            DELETE FROM entities
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM entities
                GROUP BY data_id, entity
            );
            """
            cursor.execute(query)
            conn.commit()
            print("重复记录已删除")

        except Exception as e:
            print(f"Error occurred during database insert or commit: {e}")
            
        finally:
            cursor.close()
            conn.close()


    def create_entity_link(self, kg_id):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
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
            conn.commit()  # 提交表的创建
            print("Table entity_links created or already exists")
        except Exception as e:
            print(f"Error creating table: {e}")
            return
        cursor.execute('select * from entity_links')
        result = cursor.fetchall()
        print(len(result))
        # 2. 获取一次性查询表中的数据
        cursor.execute("SELECT data_id, entity FROM entities_view WHERE type='table'")
        table_results = cursor.fetchall()
        cursor.execute("SELECT data_id, entity FROM entities_view WHERE type='image'")
        image_results = cursor.fetchall()
        cursor.execute("SELECT data_id, entity FROM entities_view WHERE type='text'")
        text_results = cursor.fetchall()

        # 将查询结果按 entity 分类
        entity_dict = {}

        for data_id, entity in table_results + image_results + text_results:
            if entity not in entity_dict:
                entity_dict[entity] = {'table': [], 'image': [], 'text': []}
            if (data_id, entity) in table_results:
                entity_dict[entity]['table'].append(data_id)
            if (data_id, entity) in image_results:
                entity_dict[entity]['image'].append(data_id)
            if (data_id, entity) in text_results:
                entity_dict[entity]['text'].append(data_id)

        # 插入数据到 entity_links 表
        for entity, data in entity_dict.items():
            # 组合不同模态的数据并插入
            for type_1, data_ids_1 in data.items():
                for type_2, data_ids_2 in data.items():
                    if type_1 != type_2 and data_ids_1 and data_ids_2:  # 确保两个模态不同且有数据
                        for data_id_1 in data_ids_1:
                            for data_id_2 in data_ids_2:
                                # 插入数据时，type_1 和 type_2 为不同的模态
                                cursor.execute('''
                                    INSERT INTO entity_links (data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (data_id_1, entity, data_id_2, entity, type_1, type_2, kg_id))
                                cursor.execute('''
                                    INSERT INTO entity_links (data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (data_id_2, entity, data_id_1, entity, type_2, type_1, kg_id))
                    conn.commit()  # 提交插入操作

        # 输出确认
        print(f"Inserted {sum(len(data['table']) + len(data['image']) + len(data['text']) for data in entity_dict.values())} links into 'entity_links'.")

        # 关闭连接
        cursor.close()
        conn.close()

        
    def create_KG_entity_link(self):
        question_data_list = self.question_processor.get_dev_information()
        data_id_dict = {}
        for idx, question_data in enumerate(question_data_list):
            text = question_data['text']
            question_id = question_data['qid']
            data_id_dict[question_id] = [question_id]
            chunks = list(chunk_text(text, self.max_tokens_per_chunk))
            for idx, chunk in enumerate(chunks):
                chunk_id = f'{data_id}_{idx}'
                data_id_dict[question_id].append(chunk_id)
            
            data_id_dict[question_id] = list(set(data_id_dict[question_id]))
            KG = KnowledgeGraph(self.kg_path)
            kg = KG.create_kg_view(data_id_dict[question_id])
            create_entity_link(idx)


    def create_kg_table(self):
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
        self.create_kg_table()
        # self.text_kg()
        # self.image_kg()
        # self.table_kg()
        self.postprocess()
        # self.create_KG_entity_link()

# kg_construct = KGConstructor()
# kg_construct.main()   

# text_processor = TextProcessor(600)
# text_processor.extract_text_id_list()
# data_processor = ManymodalQADataProcessor()
# questions = data_processor.get_dev_information()
# for i in range(3031, len(questions)):
#     question_data = questions[i]
#     if question_data['id'] == '00000000000000009844':
#         print(question_data['table'])
#         break



# 示例用法
# folder_path = "/data/cmf_dataset/ManyModalQA/table"  # 替换为你的 CSV 文件夹路径
# db_path = "/data/cmf/manymodalqa/table/ManymodalQA.db"               # 输出的 SQLite 数据库文件路径
# csv_to_sqlite(folder_path, db_path)

        

# # 示例用法
# db1_path = "/data/cmf/manymodalqa/ManymodalQA.db"  # 第一个数据库路径
# db2_path = "/data/cmf/manymodalqa/kg/kg.db"  # 第二个数据库路径
# compare_records(db1_path, db2_path)



