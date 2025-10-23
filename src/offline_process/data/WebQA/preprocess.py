import json
import sqlite3
from utils.utils import read_jsonl_file
from config.config import config_loader

import os
import argparse
import concurrent.futures
from utils.utils import write_json
from models.LLM.ModelFactory import ModelFactory
import sqlite3
from prompt.prompt_generator import TextEntitiesPrompt
from utils.proprecss_llm_answer import extract_llm_text_entities_new
from utils.utils import combine_dictionaries, write_json, simply_combine_dict
error_text_list, error_question_list = [], []
text_entities_result ={}
question_entities_result ={}

test_path = config_loader.get_dev_path('WEBQA')

# db_path = config_loader.get_database_config('MMQA')
# kg_path = config_loader.get_kg_config('MMQA')


def get_test_information():
    with open(test_path, 'r') as file:
        data = json.load(file)

    question_datas_list, image_id_list, text_id_list = [], [], []
    image_data = {}
    text_data = {}
    for question_id, question_data in data.items():
        question_data_dict = {}

        question = question_data.get("Q", "")
        question_data_dict['id'] = question_id
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
            

            text_data[snippet_id] = {}
            text_data[snippet_id] = {'id': snippet_id, "title": title, 'type':'text', 'text': fact_text, 'url': url, 'question_id': question_id}

        
        img_facts = question_data.get("img_Facts", [])
        for fact in img_facts:
            image_id = fact.get("image_id", "")
            title = fact.get('title', '')
            caption = fact.get('caption', '')
            url = fact.get('url', '')
            imgUrl = fact.get('imgUrl', '')

            facts_list.append(image_id)
            image_id_list.append(image_id)

            image_data[image_id] = {}
            image_data[image_id] = {'id': image_id, 'title': title, 'type': 'image', 'caption': caption, 'url': url, 'image_url': imgUrl, 'question_id': question_id}
        
        question_data_dict['facts_list'] = facts_list
        question_datas_list.append(question_data_dict)

    return question_datas_list, image_data, text_data, list(set(text_id_list)), list(set(image_id_list))


def llm_entities(text, text_id, args, isTitle, LLM):
    entities = {}
    max_num_tries = args.max_num_tries
    if isTitle:
        max_num_tries = 3
    count = 0
    flag = True
    # print(f'------------{text_id}------------')
    while max_num_tries:
        # print(max_num_tries)
        max_num_tries -= 1
        try:
            LLM.init_message()
            prompt_generator = TextEntitiesPrompt()
            prompt = prompt_generator.create_prompt(text)
            # print(prompt)
            result = LLM.predict(args, prompt)      
            # print(result)
            text_entities = extract_llm_text_entities_new(result)
            # print('--------text entities-------------')
            # print(text_entities)
            entities = simply_combine_dict(text_entities, entities)
        # print('-------combine entities--------------')
        # print(entities)
        except Exception as e:
            count +=1
            if count>3 and max_num_tries>0 and not entities:
                print(f"{text_id} {text} Extract entities error occurred: {e}")
                max_num_tries = 0
                flag = False
                return entities, flag
            elif count<=3:
                max_num_tries += 1
            else:
                if entities:
                    flag = True
                else:
                    flag = False
                return entities, flag
    if entities:
        flag = True
    else:
        flag = False
    return entities, flag

def extract_text_entities(text, text_id, args, isTitle, model):
    entities = {}
    entities, flag = llm_entities(text, text_id, args, isTitle, model)
    if not entities:
        flag = False
    return entities, flag

def get_texts_entities_result(text_data, args, model):#text_data:id、title、text
    text_id = text_data['id']
    title = text_data['title']
    data_type = text_data['type']
    if data_type == 'text':
        text = text_data['text']
    else:
        text = text_data['caption']
    text_entities, text_flag = extract_text_entities(text, text_id, args, False, model)
    head_entities, head_flag = extract_text_entities(title, text_id, args, True, model)
    if head_flag and text_flag:

        text_entity_names = text_entities.keys()
        head_entity_names = head_entities.keys()

        common_entity = list(set(head_entity_names).intersection(set(text_entity_names)))
        for entity in common_entity:
            head_entities.pop(entity)
        

        text_entities_result[text_id] = {}
        text_entities_result[text_id] = {"id": text_id, "type": data_type, "title": title, "text": text, "head_entities": head_entities, "fact_entities": text_entities}
        # print(text_entities_result[text_id])
        return text_entities_result[text_id], head_flag, text_flag
    else:
        return {}, head_flag, text_flag


# def process_texts_concurrently(texts, args):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#         future_to_text = {executor.submit(get_texts_entities_result, text, args): text for text in texts}
#         for future in concurrent.futures.as_completed(future_to_text):
#             text_data = future_to_text[future]
#             try:
#                 result = future.result()
#                 print(f"Text with ID {text_data['id']} generated")
#             except Exception as exc:
#                 error_text_list.append(text_data)
#                 print(f"Text with ID {text_data['id']} generated an exception: {exc}")

def write_entities_to_db(conn, entities_data, text_flag, head_flag):
    if head_flag and text_flag:
        cursor = conn.cursor()
      
        batch_insert = []
        # print('Initial batch_insert:', batch_insert)
        try:
            # Head entities
            for key, value in entities_data["head_entities"].items():
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
                        entities_data['text'], isTextTitle, isImageTitle))

            # Fact entities
            for key, value in entities_data["fact_entities"].items():  
                entity_name = key
                entity_type = value.get('Type')
                description = value.get("Description")
                batch_insert.append(
                    (entities_data["id"], entity_name, entity_type, entities_data['type'], entities_data['title'], description, 
                        entities_data['text'], 0, 0))

            # print('Final batch_insert:', batch_insert)

            # Executing batch insert
            cursor.executemany(
                'INSERT OR IGNORE INTO entities (data_id, entity, entity_type, type, title, description, text, '
                'isTextTitle, isImageTitle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_insert)

            # print('Batch insert executed successfully.')

            # Committing transaction
            conn.commit()
            # print('11111: Commit successful')
            return True, ''
        
        except Exception as e:
            print(f"Error occurred during database insert or commit: {e}")
            return False, e
        
        finally:
            cursor.close()



def process_texts_concurrently(texts, args):
    results = []

    # Connect to SQLite database
    print(f'{args.save_folder}/{args.dataset}_kg_{args.engine}.db')
    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_kg_{args.engine}.db')
    
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
        isImageTitle INTEGER DEFAULT 0
    )
    ''')
    conn.commit()

    model = ModelFactory.create_model(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_text = {executor.submit(get_texts_entities_result, text, args, model): text for text in texts}

        for future in concurrent.futures.as_completed(future_to_text):
            text_data = future_to_text[future]
            try:
                result, text_flag, head_flag = future.result()
                results.append(result)
                flag, message = write_entities_to_db(conn, result, head_flag, text_flag)
                if flag:
                # if text_flag and head_flag:
                    print(f"Text with ID {text_data['id']} processed successfully.")
                else:
                    error_text_list.append(text_data)
                    print(f"Text with ID {text_data['id']} generated an exception: {message}")
            except Exception as exc:
                error_text_list.append(text_data)
                print(f"Text with ID {text_data['id']} generated an exception: {exc}")

    # Close the database connection
    conn.close()
    return results, error_text_list

def extract_MMQA_image_table_title(args):
    questions,text_questions,table_questions,image_questions = get_dev_information()
    image_mapping = get_image_mapping()
    for i in range(args.start, args.end):
        question = questions[i]
        image_list += question["image_doc_ids"]
    image_list = list(set(image_list))

    for image_id in image_list:
        image_data = image_mapping[image_id]
        image_data_list.append(image_data)


def extract_WEBQA_text_entites(args):
    question_datas_list, image_data_mapping, text_data_mapping, text_id_list, image_id_list =  get_test_information()
    text_data_list = []
    image_data_list = []

    conn = sqlite3.connect(f'{args.save_folder}/{args.dataset}_kg_{args.engine}.db')
    cursor = conn.cursor()
    cursor.execute('select distinct(data_id) from entities')
    used_data_list = [row[0] for row in cursor.fetchall()]
    print(len(text_id_list))
    text_id_list = list(set(text_id_list) - set(used_data_list))
    print(len(text_id_list))
    print(len(image_id_list))
    image_id_list = list(set(image_id_list) - set(used_data_list))
    print(len(image_id_list))
    cursor.close()
    conn.close()
    print(f'未处理文本:{len(text_id_list)/120699}')
    print(f'未处理图像:{len(image_id_list)/90760}')


    for text_id in text_id_list:
        text_data = text_data_mapping[text_id]
        text_data_list.append(text_data)
    for image_id in image_id_list:
        image_data = image_data_mapping[image_id]
        image_data_list.append(image_data)
    
    results, error_text_list = process_texts_concurrently(text_data_list,args)
    # results, error_text_list = process_texts_concurrently(image_data_list,args)

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
        "--inference_method", default='api', help=""
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

def main(args):
    if args.dataset == 'webqa':
        extract_WEBQA_text_entites(args)
    entity_outpath = os.path.join(args.save_folder, f'{args.dataset}_text_entities_{args.engine}_new.json')
    write_json(entity_outpath, text_entities_result)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

    # args = parse_arguments()
    # text = 'William Christopher Handy (November 16, 1873 – March 28, 1958) was a composer and musician, known as the Father of the Blues. An African American, Handy was one of the most influential songwriters in the United States. One of many musicians who played the distinctively American blues music, Handy did not create the blues genre and was not the first to publish music in the blues form, but he took the blues from a regional music style (Delta blues) with a limited audience to a new level of popularity.'
    # entities = extract_text_entities(text, '1', args, 0)
    # print(entities)


    