# webqa_data_processor.py

import json
from utils.utils import read_jsonl_file
from utils.utils import write_json
from config.config import config_loader
import sqlite3
import os

def safe_upper(value):
    return value.upper() if value else value

class WebQADataProcessor:
    def __init__(self):
        self.test_path = config_loader.get_dev_path('WEBQA')
        self.image_path = config_loader.get_image_path('WEBQA')

    def get_test_information(self):
        with open(self.test_path, 'r') as file:
            data = json.load(file)

        question_datas_list, image_id_list, text_id_list = [], [], []
        image_data = {}
        text_data = {}
        for question_id, question_data in data.items():
            question_data_dict = {}
            one_text_data_list = []
            one_image_data_list = []

            question = question_data.get("Q", "")
            question_data_dict['qid'] = question_id
            question_data_dict['question'] = question

            facts_list = []
            txt_facts = question_data.get("txt_Facts", [])
            for fact in txt_facts:
                title = fact.get("title", "")
                fact_text = fact.get("fact", "")
                url = fact.get("url", "")
                snippet_id = fact.get("snippet_id", "")
                facts_list.append(snippet_id)
                text_id_list.append(snippet_id)
                text_data[snippet_id] = {'id': snippet_id, "title": title, 'type':'text', 'text': fact_text, 'url': url, 'question_id': question_id, 'question':question}
                one_text_data_list.append((snippet_id, title, fact_text))
            img_facts = question_data.get("img_Facts", [])
            for fact in img_facts:
                image_id = str(fact.get("image_id", ""))
                title = fact.get('title', '')
                caption = fact.get('caption', '')
                url = fact.get('url', '')
                imgUrl = fact.get('imgUrl', '')
                facts_list.append(image_id)
                image_id_list.append(image_id)

                image_data[image_id] = {'id': image_id, 'title': title, 'type': 'image', 'caption': caption, 'url': url, 'image_url': imgUrl, 'question_id': question_id, 'question':question}
                one_image_data_list.append((image_id, title, caption))

            question_data_dict['facts_list'] = facts_list
            question_data_dict['text_data_list'] = one_text_data_list
            question_data_dict['image_data_list'] = one_image_data_list
            question_datas_list.append(question_data_dict)

        return question_datas_list, image_data, text_data, list(set(text_id_list)), list(set(image_id_list))
class ImageProcessor:
    def __init__(self):
        self.image_path = config_loader.get_final_image('WEBQA')
        self.data_processor = WebQADataProcessor()
        self.imgae_caption_path = config_loader.get_image_path('WEBQA')
        self.object_path = '/data/cmf/webqa/image_object'
    
    def create_image_mapping(self):
        data_list = read_jsonl_file(self.imgae_caption_path)
        id_mapping = {}
        for data in data_list:
            image_id = data.get('id')
            title = data.get('title')
            path = data.get('path')
            url = data.get('url')
            caption = data.get('caption')
            path = data.get('path')
            if image_id:
                object_dict = self.get_image_object(image_id, 0.2, 0.2)
                id_mapping[image_id] = {'id': image_id, 'title': title, 'url': url, 'path': path , 'type': 'image', 'caption': caption, 'object':object_dict}
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
    
class KGConstructor:
    def __init__(self):
        self.question_processor = WebQADataProcessor()
        # self.text_processor = TextProcessor(max_tokens=600)
        self.image_processor = ImageProcessor()
        self.text_db_path = '/data/cmf/webqa/kg/webqa_kg_llama3-8b.db'
        self.image_db_path = '/data/cmf/webqa/kg/webqa_image_kg_llama3-8b.db'
        self.kg_path = '/data/cmf/webqa/kg/kg.db'
    
    
    def text_kg(self):
        db_path = self.text_db_path
        print(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = 'SELECT data_id, entity, entity_type, type, title, description, text from entities'
        cursor.execute(sql)
        entity_results = cursor.fetchall()
        text_uppercase_results = []
        image_uppercase_results = []

    
        for data_id, entity, entity_type, type_, title, description, text in entity_results:        
            text_uppercase_results.append((
            data_id, 
            safe_upper(entity), 
            safe_upper(entity_type), 
            type_, 
            safe_upper(title),
            safe_upper(description), 
            safe_upper(text)
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

        source_conn = sqlite3.connect(self.image_db_path)
        source_cursor = source_conn.cursor()

        sql = 'SELECT data_id, entity, entity_type, type, title, description, text, isImageTitle from entities'
        source_cursor.execute(sql)
        entity_results = source_cursor.fetchall()
        image_uppercase_results = []
        for data_id, entity, entity_type, type_, title, description, text, is_image_title in entity_results:        
            image_data = image_mapping[data_id]
            title = image_data['title']
            url = image_data['url']
            image_path = image_data['path']
            caption = image_data['caption']
            image_uppercase_results.append((
            data_id, 
            safe_upper(entity), 
            'image',
            safe_upper(title),
            safe_upper(caption), 
            safe_upper(caption), 
            1,
            image_path
        ))

        source_cursor.close()
        source_conn.close()



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

        cursor.executemany(entity_insert_query, image_uppercase_results)
        conn.commit()
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
                relations.append((image_id, title.upper(), entity.upper(), 'HAS', f'{title.upper()} HAS {attribute_entity.upper()}', f'{title.upper} {caption.upper()}', 'image'))

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

    def postprocess(self):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        try:
            sql = "DELETE FROM entities WHERE entity IS NULL OR entity = '' OR entity = '-';"
            cursor.execute(sql)
            conn.commit()

            sql = "DELETE FROM relationships WHERE head_entity IS NULL OR head_entity = '' OR tail_entity IS NULL OR tail_entity = '' OR relation IS NULL OR relation = '' OR head_entity = '-' OR tail_entity = '-' OR relation = '-' OR description = '-' OR description = '' OR description IS NULL;"
            cursor.execute(sql)
            conn.commit()

        except Exception as e:
            print(f"Error occurred during database insert or commit: {e}")
            
        finally:
            cursor.close()
            conn.close()
        


    def main(self):
        # self.create_kg_table()
        # print('创建完成')
        # self.text_kg()
        # print('text kg 完成')
        # self.image_kg()
        # print('image kg 完成')
        self.postprocess()



class QuestionKGConstructor:
    def __init__(self):
        self.data_processor =  WebQADataProcessor()
        self.question_kg_path = '/data/cmf/webqa/kg/kg_sqlit.db'
        self.kg_path = '/data/cmf/webqa/kg/kg.db'

    def create_kg_table(self, qid):
        conn = sqlite3.connect(self.question_kg_path)
        
        # Create the table if it does not exist
        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {qid}_entities (
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
        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {qid}_relationships (
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

    def create_question_kg(self):
        question_data_list, _, _, _, _ = self.data_processor.get_test_information()
        for question_data in question_data_list:
            data_id_list = question_data['facts_list']
            question_id = question_data['qid']
            self.create_kg_database(question_id)
            conn = sqlite3.connect(self.kg_path)
            cursor = conn.cursor()
            
            data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
            
            sql = f'SELECT data_id, entity, entity_type, type, title, text, description, isImageTitle, image_path from entities where data_id in {data_id_list_str}'
            cursor.execute(sql)
            entity_results = cursor.fetchall()
            
            sql = f'SELECT data_id, head_entity, tail_entity, relation, description, text, type, strength  from relationships where data_id in {data_id_list_str}'
            cursor.execute(sql)
            relation_results = cursor.fetchall()

            cursor.close()
            conn.close()

            conn = sqlite3.connect(self.question_kg_path)
            cursor = conn.cursor()
            
            entity_insert_query = '''
            INSERT INTO entities 
            (data_id, entity, entity_type, type, title, text, description, isImageTitle, image_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ? ,? )
        '''
            cursor.executemany(entity_insert_query, entity_results)
            conn.commit()


            relation_insert_query = '''
            INSERT INTO relationships 
            (data_id, head_entity, tail_entity, relation, description, text, type, strength) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
            cursor.executemany(relation_insert_query, relation_results)
            conn.commit()
            cursor.close()
            conn.close()



    def main(self):
        self.create_question_kg()


# kg_construct = KGConstructor()
# kg_construct.main()