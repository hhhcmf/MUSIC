import json
import io
import re
import csv
import pandas as pd
import sqlite3
from config.config import config_loader

test_path = config_loader.get_dev_path('ManymodalQA')

db_path = '/home/cmf/multiQA/data/ManymodalQA/ManymodalQA copy.db'

def extract_question_information(data):
    extracted_info = {}
    extracted_info["question"] = data.get("question", "")
    extracted_info["id"] = data.get("id", "")
    extracted_info["answer"] = data.get('answer')
    extracted_info["image"] = data.get('image')
    
    extracted_info["text"] = data.get('text')
    extracted_info['table'] = data.get('table')
    extracted_info['qtype'] = data.get('q_type')
   
    return extracted_info

def get_dev_information(test_path = test_path):
    with open(test_path, 'r') as file:
        data_list = json.load(file)
    result = []
    question_data_list = []
    for data in data_list:  
        extracted_info = extract_question_information(data)
        question_data_list.append(extracted_info)

    return question_data_list



# def create_table_from_data(data_list):
#     """根据每个数据字典的 table 字段创建表。"""
#     conn = sqlite3.connect(db_path)  # 数据库文件名
#     cursor = conn.cursor()
    
#     for data in data_list:
#         try:
#             if 'id' in data and 'table' in data and data['table'] is not None:
#                 table_name = data['id']  # 使用第一个 id 作为表名
#                 table_data = data['table']
#                 # print(table_data)
#                 lines = table_data.strip().split('\n')
#                 # print(len(lines))
                
#                 rows = []
#                 for line in lines:
#                     reader = csv.reader(io.StringIO(line))
#                     for row in reader:
#                         rows.append(row)
                        
#                 if len(rows) < 2:
#                     print(f"Table for {table_name} does not have enough data.")
#                     continue
                
#                 original_columns = [col.strip() for col in rows[0]]
#                 column_counts = {}
#                 unique_columns = []

#                 for col in original_columns:
#                     if col in column_counts:
#                         column_counts[col] += 1
#                         unique_columns.append(f"{col}{column_counts[col]}")  # 重命名为 col{i}
#                     else:
#                         column_counts[col] = 1
#                         unique_columns.append(col)

#                 columns_sql = ', '.join([f'"{col}" TEXT' for col in unique_columns])  # 使用 TEXT 类型
                
#                 # 创建表
#                 cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql});')
#                 # print(f"Table '{table_name}' created with columns: {unique_columns}")
                
#                 # 插入数据
#                 for row in rows[1:]:
#                     # 转义引号
#                     values = [val.strip().replace('"', '""') for val in row]
#                     placeholders = ', '.join(['?'] * len(values))
#                     cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', values)
                    
#                 # print(f"Data inserted into table '{table_name}'.")
#             else:
#                 flag =0 
#                 # print("Data does not contain valid 'id' or 'table' is missing.")
#         except Exception as e:
#             # print(f'{table_name} throw a exception {e}')
#             # print(data['q_type'])
#             if data['q_type'] == 'table':
#                 print(table_data)
#             # print(table_data)
#             print('-----------')
    
#     conn.commit()
#     conn.close()



def create_table_from_data(data_list):
    """根据每个数据字典的 table 字段创建表。"""
    conn = sqlite3.connect(db_path)  # 数据库文件名
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
                
                
                # 插入数据
                for idx, row in enumerate(rows[1:]):
                    # 转义引号
                    # if table_name == '00000000000000001216':
                    #     print(row)
                    #     print(lines[idx+1])
                    try:
                        values = [val.strip().replace('"', '""') for val in row]
                        placeholders = ', '.join(['?'] * len(values))
                        cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', values)
                        conn.commit()
                    except Exception as e:
                        pattern = r'"([^"]*(?:""[^"]*)*)"'
                        row = re.findall(pattern, lines[idx+1])
                        # print(row)
                        values = [val.strip().replace('"', '""') for val in row]
                        placeholders = ', '.join(['?'] * len(values))
                        cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', values)
                        
                        # # 输出结果
                        # for match in matches:
                        #     print(match)
                conn.commit() 
                # print(f"Data inserted into table '{table_name}'.")

            else:
                d=0
                # print("Data does not contain valid 'id' or 'table' is missing.")
        
        except Exception as e:
            error += 1
            # 在发生异常时使用正则表达式处理列分割
            print(f"Exception occurred for table '{table_name}': {e}")
            # print(table_data)
            # print('-----------')
    
    conn.commit()
    conn.close()
    print(error)


with open('/home/cmf/multiQA/data/ManymodalQA/data/official_aaai_split_dev_data.json', 'r') as file:
    data = json.load(file)
num = 0
for d in data:
#     if d['table']:
#         num+=1
# print(num)

    if d['id'] == '00000000000000009860':
        print(d)
        print('----------')
        table_name = d['id']
        # print(d['table'])
        
        print(d['q_type'])
        print(d['question'])
        print(d['text'])
        print(d['image'])
        print(d['answer'])


# create_table_from_data(data)

# import re

# data = '"2011","White","Red, Grey","TATA","Pirelli, Cosworth","Due to a lack of sponsorship, the team wrote various messages on the car, such as \"This could be you\", \"This is a cool spot\" and \"Your logo here\". After being purchased by Thesan Capital halfway through 2011, the messages were replaced by a silver HRT logo."'

# # 使用正则表达式匹配双引号包裹的内容
# pattern = r'"([^"]*(?:""[^"]*)*)"'
# matches = re.findall(pattern, data)

# # 输出结果
# for match in matches:
#     print(match)

# # 使用 io.StringIO 使 csv.reader 能够读取字符串
# reader = csv.reader(io.StringIO(data))

# # 打印每一行
# for row in reader:
#     print(row)
#     print(len(row))



# import csv
# import io

# # 你的 CSV 数据
# csv_data = """"Rank","Rider","Team","Time"
# "1","Cadel Evans  (AUS)","BMC Racing Team","83h 45' 20\""
# "2","Andy Schleck  (LUX)","Leopard Trek","+ 1' 34\""
# "3","Fränk Schleck  (LUX)","Leopard Trek","+ 2' 30\""
# "4","Thomas Voeckler  (FRA)","Team Europcar","+ 3' 20\""
# "5","Samuel Sánchez  (ESP)","Euskaltel–Euskadi","+ 4' 55\""
# "6","Damiano Cunego  (ITA)","Lampre–ISD","+ 6' 05\""
# "7","Ivan Basso  (ITA)","Liquigas–Cannondale","+ 7' 23\""
# "8","Tom Danielson  (USA)","Garmin–Cervélo","+ 8' 15\""
# "9","Jean-Christophe Péraud  (FRA)","Ag2r–La Mondiale","+ 10' 11\""
# "10","Pierre Rolland  (FRA)","Team Europcar","+ 10' 43\""
# """

# # 使用 io.StringIO 将字符串当作文件处理
# lines = csv_data.strip().split('\n')

# # 使用 csv.reader 处理每行
# rows = []
# for line in lines:
#     reader = csv.reader(io.StringIO(line))
#     for row in reader:
#         rows.append(row)

# # 打印结果
# for row in rows:
#     print(row)

# get_dev_information(data)

# import json
# def extract_fields_from_json(json_data):
#     # 提取所需字段
#     extracted_data = []
#     for entry in json_data:
#         extracted_entry = {
#             "text": entry.get("text"),
#             "id": entry.get("id"),
#             "question": entry.get("question"),
#             "image": entry.get("image"),            
#             "table": entry.get("table"),
#             "answer": entry.get("answer"),
#             "q_type": entry.get("q_type")
#         }
#         extracted_data.append(extracted_entry)
    
#     return extracted_data

# # 示例 JSON 数据
# json_str = '''
# [
#     {
#         "page": "Broch.html",
#         "text": "A broch (/\u02c8br\u0252x/) is an Iron Age drystone hollow-walled structure found in Scotland...",
#         "id": "00000000000000000710",
#         "question": "What is the Broch of Mousa near?",
#         "image": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Broch_of_Mousa_-_geograph.org.uk_-_2800.jpg/170px-Broch_of_Mousa_-_geograph.org.uk_-_2800.jpg",
#             "caption": "Broch of Mousa"
#         },
#         "table": null,
#         "answer": "water",
#         "q_type": "image"
#     }
# ]
# '''

# # 加载 JSON 字符串
# json_data = json.loads(json_str)

# # 提取字段
# extracted_fields = extract_fields_from_json(json_data)

# # 打印提取的数据
# for item in extracted_fields:
#     print(item)
