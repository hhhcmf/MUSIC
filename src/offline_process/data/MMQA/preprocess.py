import json
import sqlite3
from utils.utils import read_jsonl_file
from config.config import config_loader

text_path = config_loader.get_text_path('MMQA')
table_path = config_loader.get_table_path('MMQA')
image_path = config_loader.get_image_path('MMQA')
final_image_path = config_loader.get_final_image('MMQA')
dev_path = config_loader.get_dev_path('MMQA')
# db_path = config_loader.get_database_config('MMQA')
db_path = 'MMQA.db'
# kg_path = config_loader.get_kg_config('MMQA')
kg_path = '/home/cmf/multiQA/KG_LLM/kg.db'
def create_text_mapping(file_path = text_path):
    data_list = read_jsonl_file(file_path)
    id_mapping = {}
    for data in data_list:
        text_id = data.get('id')
        title = data.get('title')
        text = data.get('text')
        url = data.get('url')
        if text_id:
            id_mapping[text_id] = {'id': text_id, 'title': title, 'text': text, 'url': url, 'type':'text'}
    return id_mapping

def get_title_text_by_id(text_id, file_path = text_path):
    id_mapping = create_text_mapping(file_path)
    return id_mapping.get(text_id, {'id': None, "title": None, "text": None})

def get_image_mapping(file_path = image_path):
    data_list = read_jsonl_file(file_path)
    id_mapping = {}
    for data in data_list:
        image_id = data.get('id')
        title = data.get('title')
        path = data.get('path')
        url = data.get('url')
        if image_id:
            id_mapping[image_id] = {'id': image_id, 'title': title, 'url':url, 'path': final_image_path + path, 'type':'image'}
    return id_mapping

def get_image_by_id(image_id, file_path = image_path):
    id_mapping = get_image_mapping(file_path)
    return id_mapping.get(image_id, {'id': None, "title": None, "path": None})

def get_table_mapping(file_path = table_path):
    data_list = read_jsonl_file(file_path)
    id_mapping = {}
    for data in data_list:
        table_id = data.get('id')
        title = data.get('title')
        table = data.get('table')
        rows = data.get('table').get('table_rows')
        columns = data.get('table').get('header')
        name = data.get('table').get('table_name')
        if table_id:
            id_mapping[table_id] = {'id': table_id, 'title': title, 'table':table, 'name': name, 'rows':rows, 'columns': columns, 'type': 'table'}
    return id_mapping

# def get_table_by_id(table_id, file_path = table_path):
#     id_mapping = get_table_mapping(file_path)
#     return id_mapping.get(table_id, {'id': None,"title": None, 'name':None, 'rows': None, 'columns':None})

def get_table_by_id(id_mapping, table_id):
    return id_mapping.get(table_id, {'id': None,"title": None, 'name':None, 'rows': None, 'columns':None})

def get_table_columns(data):
    '''
    获得表中的列信息
    输入：table_information
    输出：columns_info列表(column_name, column_type)、主键列
    '''
    columns_info = []
    columns = data['columns']
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

def extract_column_values_by_id(id_mapping, table_id, file_path = table_path):
    columns_info = []

    data = get_table_by_id(id_mapping, table_id)
    headers = [col.get("column_name") for col in data['columns']]
    table_rows = data['rows']

    for row in table_rows:
        row_data = {}
        for header, cell in zip(headers, row):
            row_data[header] = cell.get("text")
        columns_info.append(row_data)
    return columns_info

def extract_column(table_information):
    headers = [col.get("column_name") for col in table_information['columns']]
    return headers

def flatten_2d_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def extract_rows(table_information):
    rows = []
    table_rows = table_information['rows']
    for row in table_rows:
        rows.append([item['text'] for item in row])
    return rows
    
def extract_column_rows(table_information):
    '''
    get table row information
    output:columns_info a list of row data
        row_data={
            <column_name>:{
                'value':<cell text>
                'alias':<alias>
            }
        }
    '''
    columns_info = []

    headers = [col.get("column_name") for col in table_information['columns']]
    table_rows = table_information['rows']

    for row in table_rows:
        row_data = {}
        for header, cell in zip(headers, row):
            row_data[header] = {}
            if header!='':
                row_data[header]['value'] = cell.get("text")
                # if 'wiki_title' in str(item)
                links = cell.get('links')
                for link in links:
                    if link.get('wiki_title'):
                        row_data[header]['wiki_title'] = link['wiki_title']
                    if link.get('url'):
                        row_data[header]['url'] = link['url']
                    
        columns_info.append(row_data)
    return columns_info

def extract_rows(table_information):
    # 提取有效的列名，即不为空字符串的列
    headers = [col.get("column_name") for col in table_information['columns'] ]

    result_rows = [
        [cell.get('text').upper() for header, cell in zip(headers, row) if header != ""]  # 只保留有有效列名的列
        for row in table_information['rows']
    ]
    
    return result_rows

#记录列名：单元格值存储，列表一个元素代表一行记录
def table_as_entities(table_information):
    columns_info = extract_column_values(table_information)
    entities = []
    for row in columns_info:
        entity = {}
        for column_name, cell_value in row.items():
            entity[column_name] = cell_value
        entities.append(entity)
    return entities

def get_table_wikititles(table_information):
    wiki_titles = []
    table_rows = table_information['rows']
    for row in table_rows:
        for item in row:
            if 'wiki_title' in str(item):
                for link in item['links']:
                    wiki_titles.append(link['wiki_title'])
    return wiki_titles


def extract_question_information(data):
    extracted_info = {}
    extracted_info["question"] = data.get("question", "")
    extracted_info["qid"] = data.get("qid", "")
    answers = data.get("answers", [])
    extracted_info["answer"] = []
    if answers:
        for answer in answers:
            if answer.get('answer'):
                extracted_info["answer"].append(answer.get('answer'))
    else:
        extracted_info["answer"] = []

    metadata = data.get("metadata", {})
    extracted_info["image_doc_ids"] = metadata.get("image_doc_ids", [])
    extracted_info["text_doc_ids"] = metadata.get("text_doc_ids", [])
    extracted_info["modalities"] = metadata.get("modalities", [])
    extracted_info["table_id"] = metadata.get("table_id", "")
    extracted_info['supporting_context'] = data.get('supporting_context')
    return extracted_info

def get_question_table_id(file_path = dev_path):
    table_id_list = []
    data_list = read_jsonl_file(file_path)
    for data in data_list:
        extracted_info = extract_question_information(data)
        if extracted_info["table_id"]!='':
            table_id_list.append(extracted_info['table_id'])
    
    return table_id_list
        

def get_dev_information(file_path = dev_path):
    result = []
    text_q = []
    table_q = []
    image_q = []
    data_list = read_jsonl_file(file_path)
    for data in data_list:
        extracted_info = extract_question_information(data)
        result.append(extracted_info)
        if extracted_info['modalities'] == ["text"]:
            text_q.append(extracted_info)
        if extracted_info['modalities'] == ["table"]:
            table_q.append(extracted_info)
        if extracted_info['modalities'] == ["image"]:
            image_q.append(extracted_info)
    return result, text_q, table_q, image_q

def get_supporting_context(file_path = dev_path):
    question_list = get_dev_information(file_path)
    supporting_id_list = []
    for question in question_list:
        supporting_context = question['supporting_context']
        supporting_id = []
        for supporting in supporting_context:
            supporting_id += [supporting['doc_id']]
            supporting_id_list.append(supporting_id)
    
    return supporting_id_list

def tablehash(kg_path = kg_path):
    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS table_hash")
    conn.commit()
    cursor.close()

    conn.execute('''
    CREATE TABLE IF NOT EXISTS table_hash (
        table_key TEXT PRIMARY KEY ,
        table_value TEXT )
    ''')
    conn.commit()

    table_hash = {}
    id_mapping = get_table_mapping()
    table_id_list = list(set(get_question_table_id()))
    insert_query = "INSERT OR IGNORE INTO table_hash (table_key, table_value) VALUES (?, ?)"
    for table_id in table_id_list:
        cursor = conn.cursor()
        result = []
        table_information = get_table_by_id(id_mapping, table_id)
        table_name = f"{table_information['title']}_{table_information['name']}"
        table_name = table_name.replace(' ', '_')
        table_hash[f'table_{table_id}'] = table_name
        table_hash[table_name] = f'table_{table_id}'
        result.append((f'table_{table_id}', table_name.upper()))
        result.append((table_name.upper(), table_id))
        column_data, key_column = get_table_columns(table_information)
        pos = 0
        for column_name, column_type in column_data:
            
            if column_name != '':
                pos+=1
                column = f'col{pos}_{table_id}'
                column_name = column_name.replace(' ', '_')
                table_hash[column] = column_name
                result.append((column, column_name.upper()))
        cursor.executemany(insert_query, result)
        conn.commit()
        cursor.close()
    
    conn.close()
    # return table_hash

def get_cols_by_table_id(table_id, kg_path = kg_path):
    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()
    sql = 'select columns from entities where data_id = ?'
    cursor.execute(sql, (table_id,))
    columns = cursor.fetchone()[0]
    columns_list = columns.split('@#')
    return columns_list
    
def get_tableId_by_name(table_name, kg_path = kg_path):
    '''
    根据sql中的table name映射到id，以便后续得到真正的tablename（table_<table_id>）和col(col<id>_<table_id>)
    '''
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

def get_tableCol_by_col(col, table_id, kg_path = kg_path):
    conn = sqlite3.connect(kg_path)
    cursor = conn.cursor()
    sql = 'SELECT table_key FROM table_hash WHERE table_value = ? AND table_key LIKE ?'
    cursor.execute(sql, (col, f'%{table_id}%'))
    table_col_name = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return table_col_name

def initialize_table_entity(entities, key, description, title, table_name, columns, is_column=0, is_column_value=0, wiki_title='', url ='', column=''):
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


def get_table_kg_data(id_mapping, table_id, entities):
    
    table_information = get_table_by_id(id_mapping, table_id)
    column_rows = extract_column_rows(table_information)
    title = table_information['title']
    name = table_information['name']
    table_name = f'{title}_{name}'.replace(' ','_')
    column_data, key_column = get_table_columns(table_information)
    column_name_list = []
    for column_name, column_type in column_data:    
        if column_name != '':
            # column_name = normalized_string(column_name)
            column_name_list.append(column_name.replace(' ', '_'))
    for column_name, column_type in column_data:
        # column_name = normalized_string(column_name)
        description = f'{column_name} is about {title} of {name}"'
        initialize_table_entity(entities, column_name.replace(' ', '_'), description, title, table_name, '@#'.join(column_name_list),1)
    for column_row in column_rows:
        # print(column_row)
        row_list = []
        for key, value in column_row.items():
            row_value = value.get('value')
            row_list.append(row_value)
        
        for key, value in column_row.items():
            
            row_value = value.get('value')
            # alias = value.get('alias') if value.get('alias') else ''
            wiki_title = value.get('wiki_title') if value.get('wiki_title') else ''
            url = value.get('url') if value.get('url') else ''
            # print('----------', row_value)
            # print(alias)
            columns = '@#'.join(column_name_list)
            row_data = '@#'.join(str(item) if item is not None else '' for item in row_list)
            # row_value = normalized_string(row_value)
            
            description = f'This is about "{name}" of "{title}" and contains informations of {row_value}, contains the following information: {columns} (informations are separated by "@#"). The details are {row_data}'
            if row_value and row_value!='' and row_value!='-' and row_value!='_' and row_value!='—':
                initialize_table_entity(entities, row_value, description, title, table_name, '@#'.join(column_name_list), 0, 1, wiki_title, url, key)

    description = f'This is about "{name}" of "{title}",which contains detailed informationis.'
    initialize_table_entity(entities, name, description, title, table_name, '@#'.join(column_name_list))
    description = f'This is about "{name}" of "{title}"'
    initialize_table_entity(entities, title, description, title, table_name, '@#'.join(column_name_list))
    return entities

def get_tables_kg_data():
    entities = {}
    id_mapping = get_table_mapping()
    table_id_list = list(set(get_question_table_id()))
    for table_id in table_id_list:
        entities = get_table_kg_data(id_mapping, table_id, entities)

def tabale2db(conn, id_mapping, table_id, table_row_count = None, postprocess_flag = 0):
    cursor = conn.cursor()
    table_information = get_table_by_id(id_mapping, table_id)
    table_name = f'table_{table_id}'
    rows = extract_rows(table_information)
    insert_error = 0#UNIQUE constraint failed
    if postprocess_flag:
        if table_row_count != len(rows):
            insert_error = 1
            print(table_id)
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            

    column_data, key_column = get_table_columns(table_information)
    
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

def tables2db(db_path=db_path):
    table_id_list = list(set(get_question_table_id()))
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
    id_mapping = get_table_mapping()
    for table_id in table_id_list:
        try:
            # print(f'正在处理 {table_id}')
            # tabale2db(conn, id_mapping, table_id) 
            tabale2db(conn, id_mapping, table_id, table_rows[f'table_{table_id}'], 1)#后处理
           
        except Exception as e:
            print(f"出错了: {table_id} {e}")

    conn.close()

# if __name__ == "__main__":
#     tablehash()
    # tables2db()
    # not in ["", "-", "—"]
    # file_path = '/home/cmf/MMQA/MMQA_texts.jsonl/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl'
    # result = get_title_text_by_id(text_id = "aead4258a97326ae15999e86d3a8aca1")
    # print(result['title'], result['text'])

    # print(get_image_by_id(image_id = '400d67f985f082d9b380de58e91e8f86'))

    # print(get_table_by_id(table_id = 'dcd7cb8f23737c6f38519c3770a6606f'))
    # tables2db()
    # table = get_table_by_id(table_id = 'dcd7cb8f23737c6f38519c3770a6606f')
    # table_keywords = extract_column(table) + flatten_2d_list(extract_rows(table)) + get_table_wikititles(table)
    # print(table_keywords)
    
    # print(get_supporting_context()[3])

    # print(get_dev_information()[0])