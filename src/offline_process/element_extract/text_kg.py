import threading
import concurrent.futures
from utils.thread.thread_local_data import ThreadLocalData
from data.WebQA.webqa_data_processor import WebQADataProcessor
from data.ManymodalQA.data_processor import TextProcessor
from data.MMQA.data_processor import MMQADataProcessor
from KG_LLM.offline_process.element_extract.kg_element_extractor import TextEntityExtractor, TextRelationExtractor
import argparse
import sqlite3
from models.LLM.ModelFactory import ModelFactory
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import gc
import torch
import time
from multiprocessing import Manager
import subprocess
import json
import psutil
multiprocessing.set_start_method('spawn', force=True)

class EntityProcessingManager:
    def __init__(self, args, max_workers=4, max_num_tries=5, gpu_ids=None):
        self.args = args
        self.max_workers = max_workers
        self.max_num_tries = max_num_tries
        self.gpu_ids = gpu_ids
        self.results = []
    
    def _get_device_id_for_process(self, index):
        """为每个进程分配一个 GPU"""
        device_id = self.gpu_ids[index % len(self.gpu_ids)]
        print(f"Assigned GPU {device_id} to process {index}")
        return device_id

    def get_bacth_datas(self, text_data_list, device_id=None):
        """在指定 GPU 上处理数据批次"""
        text_id_mapping, title_id_mapping = [],[]
        for text_data in text_data_list:
            text_id = text_data['id']
            title = text_data['title']
            data_type = text_data['type']
            if data_type == 'text':
                text = text_data['text']
            else:
                text = text_data['caption']

            text_id_mapping.append((text, text_id, text_data))
            title_id_mapping.append((title, text_id, text_data))

        if self.args.inference_method == 'api':
            return text_id_mapping, title_id_mapping, None
        
        torch.cuda.set_device(device_id)
        llm = ModelFactory.create_model(self.args, device_id)  # 加载模型到对应的 GPU
        
        return text_id_mapping, title_id_mapping, llm
    
    def _process_batch_on_gpu(self, text_id_mapping, title_id_mapping, llm, batch_size) :
        results = []
        extractor = TextEntityExtractor(self.args, llm, self.max_num_tries)

        num_batches = (len(text_id_mapping) + batch_size - 1) // batch_size  # Ensure we round up
        
        for i in range(num_batches):
            text_batch = text_id_mapping[i * batch_size: (i + 1) * batch_size]
            title_batch = title_id_mapping[i * batch_size: (i + 1) * batch_size]
            
            text_entities_results, text_datas = extractor.extract_entities_batch(text_batch, False)
            head_entities_results, text_datas = extractor.extract_entities_batch(title_batch, True)
            
            for text_id, entities in text_entities_results.items():
                if text_entities_results[text_id].get('status') == False or head_entities_results[text_id].get('status') == False:
                    continue
                else:
                    text_entities = entities
                    head_entities = head_entities_results[text_id]
                    
                    text_data = text_datas[text_id]
                    text_id = text_data['id']
                    title = text_data['title']
                    data_type = text_data['type']
                    if data_type == 'text':
                        text = text_data['text']
                        image_url = ''
                    else:
                        text = text_data['caption']
                        image_url = text_data['image_url']
                    entities_data = {"id": text_id, "type": data_type, "image_url": image_url, "title": title, "text": text, "head_entities": head_entities, "fact_entities": text_entities}
                    self.write_one_text_entities_to_db(entities_data, True, True)

            results.append((text_entities_results, head_entities_results, text_datas))

        del llm
        torch.cuda.empty_cache()

        return results
        
    
    def _process_batch_relations_on_gpu(self, text_id_mapping, title_id_mapping, llm, batch_size) :

        results = []
        extractor = TextRelationExtractor(self.args, llm, self.max_num_tries)

        num_batches = (len(text_id_mapping) + batch_size - 1) // batch_size  # Ensure we round up
        
        for i in range(num_batches):
            text_batch = text_id_mapping[i * batch_size: (i + 1) * batch_size]
            entities_batch_list = []
            for i, (text ,text_id, text_data) in enumerate(text_batch):
                conn = sqlite3.connect(f'{self.args.save_folder}/{self.args.dataset}_kg_{self.args.engine} copy.db')
                cursor = conn.cursor()
                query = "SELECT DISTINCT entity, entity_type FROM entities WHERE data_id = ?"
                cursor.execute(query, (text_id,))
                results = cursor.fetchall()
                cursor.close()
                conn.close()
                entities= {entity: entity_type for entity, entity_type in results}
                entities_batch_list.append(entities)
            title_batch = title_id_mapping[i * batch_size: (i + 1) * batch_size]
            
            text_relations_results, text_datas = extractor.extract_relations_batch(text_batch, entities_batch_list)
            
            for text_id, relations in text_relations_results.items():
                if text_relations_results[text_id].get('status') == False:
                    continue
                else:
                    text_data = text_datas[text_id]
                    text_id = text_data['id']
                    title = text_data['title']
                    data_type = text_data['type']
                    if data_type == 'text':
                        text = text_data['text']
                    else:
                        text = text_data['caption']
                    relations_data = {"id": text_id, "type": data_type, "title": title, "text": text, "relationships":relations}
                    self.write_one_text_relationships_to_db(relations_data, True)

            results.append((text_relations_results, _, text_datas))

        del llm
        torch.cuda.empty_cache()

        return results
    
    def process_batch_on_gpu(self, text_id_mapping, title_id_mapping, llm, batch_size, extract_object = 'entity') :
        print('enter process_batch_on_gpu ')

        if extract_object == 'entity':
            result = self._process_batch_on_gpu(text_id_mapping, title_id_mapping, llm, batch_size)
        else:
            result = self._process_batch_relations_on_gpu(text_id_mapping, '', llm, batch_size)
        
        return result

    def write_one_text_entities_to_db(self, entities_data, text_flag, head_flag):
        # if head_flag and text_flag:
        if text_flag:

            text_entities = entities_data['fact_entities']
            head_entities = entities_data['head_entities']
            text_entity_names = text_entities.keys()
            head_entity_names = head_entities.keys()

            common_entity = list(set(head_entity_names).intersection(set(text_entity_names)))
            for entity in common_entity:
                head_entities.pop(entity)

            conn = sqlite3.connect(f'{self.args.save_folder}/{self.args.dataset}_image_kg_{self.args.engine}.db')
            cursor = conn.cursor()
            batch_insert = []
            try:
                
                # Head entities
                conn.execute("PRAGMA busy_timeout = 3000") 

                for key, value in head_entities.items():
                    if entities_data['type'] == 'text':
                        isTextTitle = 1
                        isImageTitle = 0
                    else:
                        isImageTitle = 1
                        isTextTitle = 0
                        
                    entity_name = key
                    entity_type = value.get('Type')
                    description = value.get("Description")
                    batch_insert.append(
                        (entities_data["id"], entity_name, entity_type, entities_data['type'], entities_data['title'], description,
                            entities_data['text'], isTextTitle, isImageTitle ))

                # Fact entities
                for key, value in text_entities.items():  
                    entity_name = key
                    entity_type = value.get('Type')
                    description = value.get("Description")
                    batch_insert.append(
                        (entities_data["id"], entity_name, entity_type, entities_data['type'], entities_data['title'], description, 
                            entities_data['text'], 0, 0))

                cursor.executemany(
                    'INSERT OR IGNORE INTO entities (data_id, entity, entity_type, type, title, description, text, '
                    'isTextTitle, isImageTitle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_insert)

                conn.commit()
                print(f"Successfully inserted data for text ID {entities_data['id']}")
                return True, ''
                
            except Exception as e:
                print(f"Error occurred text ID {entities_data['id']} during database insert or commit: {e}")
                return False, e
            
            finally:
                cursor.close()
                conn.close()
    
    
    def write_one_text_relationships_to_db(self, relationships_data, text_flag):
        if text_flag:
            conn = sqlite3.connect(f'{self.args.save_folder}/{self.args.dataset}_image_kg_{self.args.engine}.db')
            cursor = conn.cursor()
            
            # 准备数据列表
            insert_data = []
            try:
                conn.execute("PRAGMA busy_timeout = 3000") 
                
                for head_entity, tails in relationships_data["relationships"].items():
                    for tail_entity, details in tails.items():
                        relation = details.get("Relation")
                        description = details.get("Description")
                        strength = details.get('Strength')
                        if isinstance(strength, str):
                            strength = int(strength)
                        insert_data.append((
                            relationships_data["id"],
                            head_entity,
                            tail_entity,
                            relation,
                            description,
                            strength,
                            relationships_data['type'],
                            relationships_data['text']
                        ))

                # 执行批量插入
                cursor.executemany('INSERT OR IGNORE INTO relationships (data_id, head_entity, tail_entity, relation, description, strength, type, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
                                insert_data)

                conn.commit()
                print(f"Successfully inserted data for text ID {relationships_data['id']}")
                return True, ''
            except Exception as e:
                print(f"Error occurred text ID {relationships_data['id']} during database insert or commit: {e}")
                return False, e
                
            finally:
                cursor.close()
                conn.close()    
    

    def process_texts_in_batches(self, texts):
        """分批次将任务分配到不同的 GPU 上"""
        num_gpus = len(self.gpu_ids)
        num_texts = len(texts)
        num_batches = num_gpus  

        # 计算基础批次大小和需要额外分配的数据数量
        base_batch_size = num_texts // num_batches
        extra = num_texts % num_batches  # 前extra个批次将分配一个额外的数据

        # 创建批次，确保批次数不超过GPU数量，多余的数据分配到前面的批次
        batches = []
        start = 0
        for i in range(num_batches):
            current_batch_size = base_batch_size + (1 if i < extra else 0)
            end = start + current_batch_size
            batch = texts[start:end]
            batches.append(batch)
            start = end

        results = []
        extract_object = self.args.extract_object
        # batch_size = len(texts) // len(self.gpu_ids)
        # batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # if len(texts) % num_gpus != 0:
        #     for i in range(len(texts) % num_gpus):
        #         batches[i % num_gpus].append(texts[-(i+1)])

        # results = []
        
        results = []
        total_key_value_pairs = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for index, batch_data in enumerate(batches):
                device_id = self._get_device_id_for_process(index)
                text_id_mapping, title_id_mapping, llm = self.get_bacth_datas(batch_data, device_id)
                # future = executor.submit(self._process_batch_on_gpu, text_id_mapping, title_id_mapping, llm, batch_size=1)
                future = executor.submit(self.process_batch_on_gpu, text_id_mapping, title_id_mapping, llm, batch_size=8, extract_object = extract_object)
                futures.append(future)

            # 等待所有任务完成并获取结果
            for future in concurrent.futures.as_completed(futures):
                # try:
                batch_result = future.result() 
                print(len(batch_result))
                # for result in batch_result:
                #     text_entities_results = result[0]
                #     head_entities_results = result[1]
                #     text_datas = result[2]
                #     for text_id, entities in text_entities_results.items():
                #         if text_entities_results[text_id].get('status') == False or head_entities_results[text_id].get('status') == False:
                #             continue
                #         else:
                #             text_entities = entities
                #             head_entities = head_entities_results[text_id]
                            
                #             text_data = text_datas[text_id]
                #             text_id = text_data['id']
                #             title = text_data['title']
                #             data_type = text_data['type']
                #             if data_type == 'text':
                #                 text = text_data['text']
                #             else:
                #                 text = text_data['caption']
                #             entities_data = {"id": text_id, "type": data_type, "title": title, "text": text, "head_entities": head_entities, "fact_entities": text_entities}
                #             self.write_one_text_entities_to_db(entities_data, True, True)

                    # total_key_value_pairs += len(text_entities_results)
                    # total_key_value_pairs += len(head_entities_results)
                
                results.extend(batch_result)
                # except Exception as e:
                #     print(f"Error occurred: {e}")

        return results, total_key_value_pairs

    def _process_text_entities_by_api(self, text_id_map, title_id_map, llm):
        
        extractor = TextEntityExtractor(self.args, llm, self.max_num_tries)
        
        text_data = text_id_map[2]

        text_entities_results, text_flag= extractor.extract_entities(text_id_map[0], text_id_map[1], False)
        head_entities_results, head_flag = extractor.extract_entities(title_id_map[0], title_id_map[1], True)

        text_id = text_data['id']
        title = text_data['title']
        data_type = text_data['type']
        if data_type == 'text':
            text = text_data['text']
            image_url = ''
        else:
            text = text_data['caption']
            image_url = text_data['image_url']
        


        entities_data = {"id": text_id, "type": data_type, "image_url": image_url, "title": title, "text": text, "head_entities": head_entities_results, "fact_entities": text_entities_results}
        print(f'正在处理{text_id}, {text}, {title}, {text_flag}, {head_flag}')
        self.write_one_text_entities_to_db(entities_data, text_flag, head_flag)
        return entities_data
        
        # return (text_entities_results, text_flag, head_entities_results, head_flag, text_data)

    def _process_text_relations_by_api(self, text_id_map, title_id_map, llm):
        
        extractor = TextRelationExtractor(self.args, llm, self.max_num_tries)
        
        text_data = text_id_map[2]
        text_id = text_data['id']
        conn = sqlite3.connect(f'{self.args.save_folder}/{self.args.dataset}_kg_{self.args.engine} copy.db')
        cursor = conn.cursor()
        query = "SELECT DISTINCT entity, entity_type FROM entities WHERE data_id = ?"
        cursor.execute(query, (text_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        entities= {entity: entity_type for entity, entity_type in results}

        text_relations_results, text_flag= extractor.extract_relations(text_id_map[0], text_id_map[1],entities)

        text_id = text_data['id']
        title = text_data['title']
        data_type = text_data['type']
        if data_type == 'text':
            text = text_data['text']
        else:
            text = text_data['caption']

        relations_data = {"id": text_id, "type": "text", "title": title, "text": text, "relationships":text_relations_result}

        self.write_one_text_relationships_to_db(relations_data, text_flag)

        return relations_data
        
    
    def _process_text_by_api(self, text_id_map, title_id_map, llm, extract_object = 'entity'):
        if extract_object == 'entity':
            return self._process_text_entities_by_api(text_id_map, title_id_map, llm)
        else:
            return self._process_text_relations_by_api(text_id_map, title_id_map, llm)
        

    def multithread_process_texts(self, texts):
        results = []
        # total_key_value_pairs=0
        llm = ModelFactory.create_model(self.args)
        extract_object = self.args.extract_object
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_text = {}
            futures = []
            text_id_mapping, title_id_mapping, _ = self.get_bacth_datas(texts)
            
            for text_id_map, title_id_map in zip(text_id_mapping, title_id_mapping):
                future = executor.submit(self._process_text_by_api, text_id_map, title_id_map, llm, extract_object)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result() 
                results.append(future)
        
        return results
            

def create_kg_database(args):
    # Connect to SQLite database
    print(f'{args.save_folder}/{args.dataset}_image_kg_{args.engine}.db')
    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_image_kg_{args.engine}.db')
    
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
        isTextTitle INTEGER DEFAULT 0, 
        isImageTitle INTEGER DEFAULT 0,
        image_url TEXT
    )
    ''')
    conn.commit()    

    conn.execute('''
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_id TEXT ,
        head_entity TEXT ,
        tail_entity TEXT,
        relation TEXT,
        description TEXT,
        strength INT, 
        type TEXT,
        text TEXT
    )
    ''')
    conn.commit()
    conn.close() 

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temperature", type=float, default=0.2, help=""
    )
    parser.add_argument(
        '--dataset', default='webqa',
        help="dataset",
        choices=["mmqa", "webqa","mmcovqa" ]
    )
    
    parser.add_argument(
        "--engine", default='llama3-8b', help="llama2-7b, llama3-8b, gpt-3.5-turbo, qwen-turbo, qwen-plus",
        choices=["llama2-7b", "llama3-8b", "gpt-3.5-turbo", 'qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-plus-latest']
    )
    parser.add_argument(
        '--inference_method', default='api', help=""
    )
    parser.add_argument(
        '--extract_object', default='entity', help=""
    )
    parser.add_argument(
        "--model_path", default='/home/cmf/era-cot-main/Llama-2-7b-chat-hf', help="your local model path"
    )
    parser.add_argument(
        "--api_key", default='sk-llqpW9khKdoFanaN1e992c3077Bd4a05A9D581BcA382F889', help="your api_key"
    )
    parser.add_argument(
        "--api_base", default='https://api.xiaoai.plus/v1', help="api_base"
    )
    parser.add_argument(
        "--start", type=int, default=0, help='number'
    )
    parser.add_argument(
        "--end",  type=int, default=7540, help='number, max7540'
    )
    parser.add_argument(
        '--max_num_tries',  type=int, default=5, help="llm max tries"
    )
    parser.add_argument(
        '--max_workers',  type=int, default=5, help="max workers num"
    )
    parser.add_argument(
        '--save_folder', default='/data/cmf/webqa/kg', help="save_folder"
    )
    parsed_args = parser.parse_args()
    return parsed_args

def extract_webqa_entities(args):
    data_processor = WebQADataProcessor()
    question_datas_list, image_data_mapping, text_data_mapping, text_id_list, image_id_list =  data_processor.get_test_information() 
    if args.extract_object == 'entity':
            table_name = 'entities'
    else:
        table_name = 'relationships'
    
    print(f'正在处理:{table_name}')

    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_image_kg_{args.engine}.db')
    cursor = conn.cursor()
    
    cursor.execute(f'select distinct(data_id) from {table_name}')
    used_data_list = [row[0] for row in cursor.fetchall()]

    # print(len(text_id_list))
    # text_id_list = list(set(text_id_list) - set(used_data_list))

    # print(len(text_id_list))
    print('all:',len(list(set(image_id_list))))
    print('已处理',len(list(set(used_data_list))))
    print(type(used_data_list[0]))
    image_id_list = list(set(image_id_list) - set(used_data_list))
    print('未处理',len(image_id_list))
    cursor.close()
    conn.close()
    
    image_data_list = []
    # for text_id in text_id_list:
    #     text_data = text_data_mapping[text_id]
    #     text_data_list.append(text_data)
    for image_id in image_id_list:
        image_data = image_data_mapping[image_id]
        image_data_list.append(image_data)

    # text_data_list = sorted(text_data_list, key=lambda x: x['id'])
    image_data_list = sorted(image_data_list, key=lambda x: x['id'])
    return image_data_list

def extract_mmqa_entities(args):
    data_processor = MMQADataProcessor()
    questions,text_questions,table_questions,image_questions = data_processor.get_dev_information()
    texts_mapping = data_processor.create_text_mapping()
    text_list = []
    text_data_list = []

    for question in questions:
        text_list += question['text_doc_ids']

    text_list = list(set(text_list))
    print(len(text_list))

    if args.extract_object == 'entity':
            table_name = 'entities'
    else:
        table_name = 'relationships'

    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_kg_{args.engine}.db')
    cursor = conn.cursor()

    cursor.execute(f'select distinct(data_id) from {table_name}')
    used_data_list = [row[0] for row in cursor.fetchall()]
    text_list = list(set(text_list) - set(used_data_list))
    cursor.close()
    conn.close()
    
    for text_id in text_list:
        text_data = texts_mapping[text_id]
        text_data_list.append(text_data)
    
    return text_data_list

def extract_manymodalqa_entities(args):
    text_processor = TextProcessor(args.max_tokens_per_chunk)
    question_datas_list = text_processor.data_processor.get_dev_information()
    data_list = []
    data_id_list = []
    data = {}
    for question_data in question_datas_list:
        data_id = question_data['id']

        if question_data['image']:
            # data_list.append({'id':data_id, 'image':question_data['image']})
            data_id_list.append(data_id)
            data[data_id] = {'id':data_id, 'image':question_data['image'], 'type': 'image', 'caption': question_data['image']['caption'], 'image_url': question_data['url']}

        text = question_data['text']
        chunks = list(text_processor.chunk_text(text, args.max_tokens_per_chunk))
        for idx, chunk in enumerate(chunks):
            chunk_id = f'{data_id}_{idx}'
            # data_list.append({'id':chunk_id, 'text':chunk, ''})
            data_id_list.append(chunk_id)
            data[chunk_id] ={}
            data[chunk_id] = {'id':chunk_id, 'text':chunk, 'type': 'text'}
            
    all_data = list(set(data_id_list))

    if args.extract_object == 'entity':
            table_name = 'entities'
    else:
        table_name = 'relationships'

    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_kg_{args.engine}.db')

    cursor = conn.cursor()
    cursor.execute(f'select distinct(data_id) from {table_name}')
    used_data_list = [row[0] for row in cursor.fetchall()]
    data_id_list = list(set(data_id_list) - set(used_data_list))
    cursor.close()
    conn.close()
    for d_id in data_id_list:
        text_data = data[d_id]
        data_list.append(text_data)
    print(f'未处理:{len(data_list)},已处理{len(all_data)/len(data_list)}')
    return data_list

def extract_text_entities(args):
    if args.dataset.lower() == 'webqa':
        return extract_webqa_entities(args)

    elif args.dataset.lower() == 'manymodalqa':
        return extract_manymodalqa_entities(args)
        
    elif args.dataset.lower() == 'mmqa':
        return extract_mmqa_entities(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

       
    
# 使用示例
if __name__ == "__main__":
    args = parse_arguments() 
    print(args.dataset)
    # 假设你有这些文本数据
    all_texts = extract_text_entities(args)
    
    # texts =[{'id': 'd5d6f4a80dba11ecb1e81171463288e9_5', 'title': 'Telegraph Creek Formation', 'type': 'text', 'text': 'The Telegraph Creek Formation is a Mesozoic geologic formation in Montana, United States. Dinosaur remains are among the fossils that have been recovered from the formation, although none have yet been referred to a specific genus.', 'url': 'https://en.wikipedia.org/wiki/Telegraph_Creek_Formation', 'question_id': 'd5d6f4a80dba11ecb1e81171463288e9'}, {'id': 'd5dee30c0dba11ecb1e81171463288e9_16', 'title': 'Wegener (lunar crater)', 'type': 'text', 'text': "Wegener is a lunar impact crater that is located in the Moon's northern hemisphere, about midway between the equator and the north pole.", 'url': 'https://en.wikipedia.org/wiki/Wegener_%28lunar_crater%29', 'question_id': 'd5dee30c0dba11ecb1e81171463288e9'}, {'id': 'd5d570060dba11ecb1e81171463288e9_10', 'title': 'Elf (Middle-earth)', 'type': 'text', 'text': 'Without Elwë, many of the Teleri took his brother Olwë as their leader and were ferried to Valinor. Some Teleri stayed behind though, still looking for Elwë, and others stayed on the shores, being called by Ossë.', 'url': 'https://en.wikipedia.org/wiki/Elf_(Middle-earth)', 'question_id': 'd5d570060dba11ecb1e81171463288e9'}, {'id': 'd5ca2a980dba11ecb1e81171463288e9_0', 'title': "St Mark's Campanile", 'type': 'text', 'text': 'The tower is capped by a pyramidal spire at the top of which there is a golden weather vane in the form of the archangel Gabriel.', 'url': "https://en.wikipedia.org/wiki/St_Mark's_Campanile", 'question_id': 'd5ca2a980dba11ecb1e81171463288e9'}, {'id': 'd5d19d8c0dba11ecb1e81171463288e9_13', 'title': 'Neil Finn - Wikipedia', 'type': 'text', 'text': 'Neil Mullane Finn OBE (born 27 May 1958) is a New Zealand singer-songwriter and musician who is currently a member of Fleetwood Mac. With his brother Tim Finn, he was the co- frontman for Split Enz, a project that he joined after it was initially founded by Tim and others, and then became the frontman for Crowded House.', 'url': 'https://en.wikipedia.org/wiki/Neil_Finn', 'question_id': 'd5d19d8c0dba11ecb1e81171463288e9'}, {'id': 'd5cf63a00dba11ecb1e81171463288e9_8', 'title': 'Morgen (mythological creature) - Wikipedia', 'type': 'text', 'text': 'Morgens, morgans, or mari-morgans are Welsh and Breton water spirits that drown men. The name may derive from Mori-genos or Mori-gena, meaning "sea-born. The name has also been rendered as Muri-gena or Murigen.', 'url': 'https://en.wikipedia.org/wiki/Morgen_(mythological_creature)', 'question_id': 'd5cf63a00dba11ecb1e81171463288e9'}, {'id': 'd5da50c60dba11ecb1e81171463288e9_6', 'title': 'Snake', 'type': 'text', 'text': 'In such a show, the snake charmer carries a basket containing a snake that he seemingly charms by playing tunes with his flutelike musical instrument, to which the snake responds.', 'url': 'https://en.wikipedia.org/wiki/Snake', 'question_id': 'd5da50c60dba11ecb1e81171463288e9'}, {'id': 'd5c1d3d40dba11ecb1e81171463288e9_13', 'title': 'Hong Kong Disneyland', 'type': 'text', 'text': "However, it was announced in September 2020 by the Hong Kong Government that Hong Kong Disneyland's option to purchase the 60-hectare expansion site next to the existing park will not be extended after its expiry on 24 September 2020 as it is unable to commit to using the site in the near future.", 'url': 'https://en.wikipedia.org/wiki/Hong_Kong_Disneyland', 'question_id': 'd5c1d3d40dba11ecb1e81171463288e9'}, {'id': 'd5dee65e0dba11ecb1e81171463288e9_3', 'title': 'Equation of time', 'type': 'text', 'text': "The right ascension, and hence the equation of time, can be calculated from Newton's two-body theory of celestial motion, in which the bodies (Earth and Sun) describe elliptical orbits about their common mass center.", 'url': 'https://en.wikipedia.org/wiki/Equation_of_time', 'question_id': 'd5dee65e0dba11ecb1e81171463288e9'}, {'id': 'd5c7e2420dba11ecb1e81171463288e9_5', 'title': 'Northgate High School (Walnut Creek, California)', 'type': 'text', 'text': 'Northgate High School (NHS) is a public high school located in the suburban Northgate neighborhood of Walnut Creek, California, United States. The most recent of five high schools in the Mount Diablo Unified School District, the school was built in 1974, and is home to approximately 1,500 students from Walnut Creek and Concord, California, grades 9–12.', 'url': 'https://en.wikipedia.org/wiki/Northgate_High_School_(Walnut_Creek,_California)', 'question_id': 'd5c7e2420dba11ecb1e81171463288e9'}, {'id': 'd5de3e3e0dba11ecb1e81171463288e9_11', 'title': 'Weight training', 'type': 'text', 'text': 'Different types of weights will give different types of resistance, and often the same absolute weight can have different relative weights depending on the type of equipment used.', 'url': 'https://en.wikipedia.org/wiki/Weight_training', 'question_id': 'd5de3e3e0dba11ecb1e81171463288e9'}, {'id': 'd5d0fce20dba11ecb1e81171463288e9_14', 'title': 'County (United States)', 'type': 'text', 'text': "For example, Gwinnett County, Georgia, and its county seat, the city of Lawrenceville, each have their own police departments. (A separate county sheriff's department is responsible for security of the county courts and administration of the county jail.)", 'url': 'https://en.wikipedia.org/wiki/County_(United_States)', 'question_id': 'd5d0fce20dba11ecb1e81171463288e9'}, {'id': 'd5d3bc160dba11ecb1e81171463288e9_12', 'title': 'Dolph Lundgren - Wikipedia', 'type': 'text', 'text': "Hans Lundgren ( / ˈlʌndɡrən /, Swedish: [ˈdɔlːf ˈlɵ̌nːdɡreːn] ( listen); born 3 November 1957), better known as Dolph Lundgren, is a Swedish actor, filmmaker, and martial artist. Lundgren's breakthrough came in 1985, when he starred in Rocky IV as the imposing Soviet boxer Ivan Drago.", 'url': 'https://en.wikipedia.org/wiki/Dolph_Lundgren', 'question_id': 'd5d3bc160dba11ecb1e81171463288e9'}, {'id': 'd5d0643a0dba11ecb1e81171463288e9_12', 'title': 'Muhammad Khan Sherani', 'type': 'text', 'text': 'He was re-elected to the National Assembly as a candidate of Muttahida Majlis-e-Amal (MMA) from Constituency NA-264 (Zhob-cum-Killa Saifullah) in 2002 Pakistani general election.', 'url': 'https://en.wikipedia.org/wiki/Muhammad_Khan_Sherani', 'question_id': 'd5d0643a0dba11ecb1e81171463288e9'}, {'id': 'd5d867200dba11ecb1e81171463288e9_9', 'title': 'The Matrix', 'type': 'text', 'text': "Davis' score combines orchestral, choral and synthesizer elements; the balance between these elements varies depending on whether humans or machines are the dominant subject of a given scene.", 'url': 'https://en.wikipedia.org/wiki/The_Matrix', 'question_id': 'd5d867200dba11ecb1e81171463288e9'}, {'id': 'd5bc16ba0dba11ecb1e81171463288e9_6', 'title': "Gymnastics at the 2014 Summer Youth Olympics – Boys' artistic qualification", 'type': 'text', 'text': "Boys' artistic gymnastics qualification at the 2014 Summer Youth Olympics was held at the Nanjing Olympic Sports Centre on August 17. The results of the qualification determined the qualifiers to the finals: 18 gymnasts in the all-around final, and 8 gymnasts in each of 4 apparatus finals.", 'url': "https://en.wikipedia.org/wiki/Gymnastics_at_the_2014_Summer_Youth_Olympics_–_Boys'_artistic_qualification", 'question_id': 'd5bc16ba0dba11ecb1e81171463288e9'}, {'id': 'd5d0e6940dba11ecb1e81171463288e9_14', 'title': 'Languages of Hong Kong', 'type': 'text', 'text': 'Vietnamese is used in Hong Kong among the ethnic Chinese from Vietnam who had initially settled in Vietnam and returned to Hong Kong. The language is also used by Vietnamese refugees who left their home during the Vietnam War.', 'url': 'https://en.wikipedia.org/wiki/Languages_of_Hong_Kong', 'question_id': 'd5d0e6940dba11ecb1e81171463288e9'}, {'id': 'd5da6b4c0dba11ecb1e81171463288e9_13', 'title': 'Pathophysiology of acute respiratory distress syndrome - Wikipedia', 'type': 'text', 'text': 'The pathophysiology of acute respiratory distress syndrome involves fluid accumulation in the lungs not explained by heart failure (noncardiogenic pulmonary edema).', 'url': 'https://en.wikipedia.org/wiki/Pathophysiology_of_acute_respiratory_distress_syndrome', 'question_id': 'd5da6b4c0dba11ecb1e81171463288e9'}, {'id': 'd5dccb940dba11ecb1e81171463288e9_15', 'title': 'Ghaznavids - Wikipedia', 'type': 'text', 'text': 'In 997, Mahmud, another son of Sebuktigin, succeeded to the throne, and Ghazni and the Ghaznavid dynasty have become perpetually associated with him. He completed the conquest of the Samanid and Shahi territories, including the Ismaili Kingdom of Multan, Sindh, as well as some Buwayhid territory.', 'url': 'https://en.wikipedia.org/wiki/Ghaznavids', 'question_id': 'd5dccb940dba11ecb1e81171463288e9'}, {'id': 'd5c52e620dba11ecb1e81171463288e9_6', 'title': 'Eurasian oystercatcher', 'type': 'text', 'text': 'The Eurasian oystercatcher (Haematopus ostralegus) also known as the common pied oystercatcher, or palaearctic oystercatcher, or (in Europe) just oystercatcher, is a wader in the oystercatcher bird family Haematopodidae. It is the most widespread of the oystercatchers, with three races breeding in western Europe, central Eurosiberia, Kamchatka, China, and the western coast of Korea.', 'url': 'https://en.wikipedia.org/wiki/Eurasian_oystercatcher', 'question_id': 'd5c52e620dba11ecb1e81171463288e9'}, {'id': 'd5debfb20dba11ecb1e81171463288e9_4', 'title': 'Airstream mechanism', 'type': 'text', 'text': 'Consonants may be pronounced without any airstream mechanism. These are percussive consonants, where the sound is generated by one organ striking another.', 'url': 'https://en.wikipedia.org/wiki/Airstream_mechanism', 'question_id': 'd5debfb20dba11ecb1e81171463288e9'}, {'id': 'd5c124b60dba11ecb1e81171463288e9_14', 'title': 'Glossary of botanical terms', 'type': 'text', 'text': 'A fine white or bluish waxy powder occurring on plant parts, usually stems, leaves, and fruits. It is easily removed by rubbing.', 'url': 'https://en.wikipedia.org/wiki/Glossary_of_botanical_terms', 'question_id': 'd5c124b60dba11ecb1e81171463288e9'}, {'id': 'd5c7233e0dba11ecb1e81171463288e9_3', 'title': 'Suzuki Vitara', 'type': 'text', 'text': 'The second generation was launched in 1998 under the "Grand Vitara" badge in most markets. It was accompanied by a still larger SUV known as the Suzuki XL-7 (known as Grand Escudo in Japan).', 'url': 'https://en.wikipedia.org/wiki/Suzuki_Vitara', 'question_id': 'd5c7233e0dba11ecb1e81171463288e9'}, {'id': 'd5cb16f60dba11ecb1e81171463288e9_13', 'title': 'Woman with a Pearl Necklace in a Loge', 'type': 'text', 'text': 'It shows that "\'The ages of woman\'": infancy, childhood, youth or coming of age, adulthood and maternity, maturity, and old age" are a prevalent theme in each of her pieces.', 'url': 'https://en.wikipedia.org/wiki/Woman_with_a_Pearl_Necklace_in_a_Loge', 'question_id': 'd5cb16f60dba11ecb1e81171463288e9'}]
    gpu_ids = [3,7,4,5,6]
    create_kg_database(args)

    entity_manager = EntityProcessingManager(args, args.max_workers, args.max_num_tries, gpu_ids)
    print('start')
    start_time = time.time()

    # for texts in texts_batch:
    if args.inference_method == 'api':
        results = entity_manager.multithread_process_texts(all_texts)
    else:
        results, total_key_value_pairs = entity_manager.process_texts_in_batches(all_texts)
        # results, total_key_value_pairs = entity_manager.multithread_process_texts(texts)
    
    print("Processed time:")
    print(time.time()-start_time)
    # print("Processed results:", results)
    # print("Total number of entities(key-value pairs):", total_key_value_pairs)
    
    # output_file = '/home/cmf/multiQA/data/WebQA/test_entity.json'
    # with open(output_file, 'w', encoding='utf-8') as f:
    # # 使用 json.dump() 将 results 列表保存为 JSON 格式
    #     json.dump(results, f, ensure_ascii=False, indent=4)
