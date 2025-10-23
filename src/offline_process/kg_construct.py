import sqlite3
import os
import pandas as pd
from data.WebQA.preprocess import get_test_information
from data.preprocess.load_data import load_data, load_KG
from phase.kg_construct.kg import *
question_datas_list, image_data, text_data, text_id_list, image_id_list = get_test_information()
#TODO 图像标题、文本标题加入
def create_kg_table(kg_path):
    conn = sqlite3.connect(kg_path)
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
def text_kg():
    db_path = 'kg.db'
    source_db_path = '/data/cmf/webqa/kg/webqa_kg_llama3-8b.db'
    conn = sqlite3.connect(source_db_path)
    cursor = conn.cursor()
    sql = 'SELECT data_id, entity, entity_type, type, title, description, text from entities'
    cursor.execute(sql)
    entity_results = cursor.fetchall()
    uppercase_results = []
    data_id_list = []
    for data_id, entity, entity_type, type_, title, description, text in entity_results:
        data_id_list.append(data_id)
    data_id_list = set(list(data_id_list))

    for data_id, entity, entity_type, type_, title, description, text in entity_results:
        uppercase_results.append((data_id, entity.upper(), entity_type.upper(), type_, title.upper(), description.upper(), text.upper()))

    sql = 'SELECT data_id, head_entity, tail_entity, relation, description, text, type from relationships'
    cursor.execute(sql)
    relation_results = cursor.fetchall()
    cursor.close()
    conn.close()
    uppercase_relation_results = [
    (data_id, head_entity.upper(), tail_entity.upper(), relation.upper(), description.upper(), text.upper(), type_)
    for data_id, head_entity, tail_entity, relation, description, text, type_ in relation_results
]

    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()
    entity_insert_query = '''
    INSERT INTO entities 
    (data_id, entity, entity_type, type, title, description, text) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
'''
    cursor.executemany(entity_insert_query, uppercase_results)
    conn.commit()

    relation_insert_query = '''
    INSERT INTO relationships 
    (data_id, head_entity, tail_entity, relation, description, text, type) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
'''
    cursor.executemany(relation_insert_query, uppercase_relation_results)
    conn.commit()
    cursor.close()
    conn.close()

def table_kg():
    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()

    table_id_list = list(set(get_question_table_id()))
    insert_query = '''
    INSERT INTO entities 
    (data_id, entity, type, title, description, isColumn, isColumnValue, table_name, columns, column, wiki_title, url) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    id_mapping = get_table_mapping()
    for table_id in table_id_list:
        entities = {}
        result = []
        entities = get_table_kg_data(id_mapping, table_id, entities)
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
    cursor.close()
    conn.close()


def initialize_image_entity(entities, key, title, description, text, isImageTitle, url):
    key = key.upper()
    entities[key] = {
        'title': title.upper(),
        'description': description.upper(),
        'text': text.upper(),
        'isImageTitle': isImageTitle,
        'url':url
    }
img_attr_folder_path = '/home/cmf/multiQA/data/MMQA/data/imgae_caption/new'
def get_img_entity(image_id, title, url, entity_conf_value, attribute_conf_value):
    entities = {}
    relations = []
    result_dict = {}
    file_path = os.path.join(img_attr_folder_path, f'{image_id}_llava1.5.txt' )
    with open(file_path, 'r', encoding='utf-8') as file:
        caption = file.read()

    file_path = os.path.join(img_attr_folder_path, f'{image_id}.txt' )
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i in range(0, len(lines), 2): 
        attributes_line = lines[i].strip()
        confidence_line = lines[i+1].strip()
        attributes_line_list = attributes_line.split(' ')
        attributes = attributes_line_list[0]
        entity = ' '.join(attributes_line_list[1:])
        attribute_list = attributes.split(',')
        
        conf_values = list(map(float, confidence_line.replace(',', ' ').split()))
        entity_conf = conf_values[-1]

        if entity_conf> entity_conf_value:
            filtered_attributes = [attr for attr, conf in zip(attribute_list, conf_values[:-1]) if conf > attribute_conf_value]

            if entity in result_dict:
                result_dict[entity].extend(filtered_attributes)
            else:
                result_dict[entity] = filtered_attributes
    initialize_image_entity(entities, title, title,  f'{title} describes that {caption}', f'{title} describes that {caption}', 1, url)
    for entity in result_dict:
        attributes_str = ','.join(list(set(result_dict[entity])))
        # entity = f'{attributes_str} {entity}'
        attribute_entity = f'{attributes_str} {entity}'
        initialize_image_entity(entities, entity, title, f'{title} has {attribute_entity}', f'{title} describes that {caption}', 0, url)
        relations.append((image_id, title.upper(), entity.upper(), 'HAS', f'{title.upper()} HAS {attribute_entity.upper()}', f'{title.upper()} DESCRIBES {caption.upper()}', 'image'))
    return entities, relations




def image_kg(entity_conf_value=0.2, attribute_conf_value=0.2):
    questions, image_q, table_q, image_q = get_dev_information()
    image_mapping = get_image_mapping()
    image_list = []
    image_data_list = []
    entity_result = []


    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()
    entity_insert_query = '''
    INSERT INTO entities 
    (data_id, entity, type, title, description, text, isImageTitle) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''

    relation_insert_query = '''
    INSERT INTO relationships 
    (data_id, head_entity, tail_entity, relation, description, text, type) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''

    for image_id in image_id_list:
        entity_result = []
        image_data = image_mapping[image_id]
        title = image_data['title']
        url = image_data['url']
        entities, relation_results = get_img_entity(image_id, title, url, entity_conf_value, attribute_conf_value)
        # print(image_id, len(entities.items()))
        for entity_name, entity_values in entities.items():
            title = entity_values['title']
            description = entity_values['description']
            text = entity_values['text']
            isImageTitle = entity_values['isImageTitle']
            entity_result.append((image_id, entity_name, 'image', title, description, text, isImageTitle))
        # print(image_id, len(entity_result))
        cursor.executemany(entity_insert_query, entity_result)
        conn.commit()
        cursor.executemany(relation_insert_query, relation_results)
        conn.commit()
    cursor.close()
    conn.close()    

def create_KG_entity_link(start, end):
    questions,text_q, table_q, image_q = get_dev_information()
    for idx in range(start, end):
        data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
        KG = KnowledgeGraph(kg_path)
        kg = KG.create_kg_view(data_id_list)
        KG.create_entity_link(idx)


    
if __name__ == "__main__":
    create_kg_table()
    text_kg()
    table_kg()
    image_kg()
    create_KG_entity_link()



    