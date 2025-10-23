import json
import logging
import re
import os
import argparse
import asyncio
from utils.utils import print_exp, save_result, write_json, append_to_json_file
from utils.log import ThreadLogging
from data.preprocess.load_data import load_data, load_data_new, load_KG
from models.llama2_predict import predict, model_init
from models.LLM.ModelFactory import ModelFactory
from KG_LLM.vector_store.lancedb import *
from KG_LLM.online_query.agent.KG.inference import ReasoningPhase
from KG_LLM.online_query.agent.KG.KGReasonProcessor import KGReasonProcessor
from KG_LLM.online_query.base import BaseConfig, KgConfig, RetrieveConfig
from KG_LLM.KGE.retrieve import KGERetrieval, KGTrainer
from KG_LLM.online_query.retrieve import RetrieveEntitiesPhase

api_key_list = ['sk-fvXDuNXlHri5Mh02WA4HWpFHPFxktboNCEDU1wD1jGfidpqh', 'sk-egtkFOSYvqtwdoRXRe90KuorfrVR878XoUQjaFhloVDKQLW5', 'sk-iTzPg97AIoOE6xZidJsgEN3LSpdvS8gl0M4eWj28ODKF6lAt', 'sk-uxk3DmlL88wZ7ZOAgLekSVxLycK8141PE67Cooap5pB28o6a', 'sk-lt0X3GAYLtEBAEdL8reGf5VrRA9tPqu5BktkVk4kY2Wub10g', 'sk-5BVPi2najN3sNm4ktmFMrrgvTV7BT395qazqyygK4VQAGptS', 'sk-zbBDn6MpraDxOvrPtJbYFBIgm0Dris3vVaxgP1mO7NHmAsrZ', 'sk-hsWI0VhPKsKHcjjvT0pNDczpu5XjGvUA5llpZhew0VFGR2LW', 'sk-2mPW41MgwpU2C8mQ9chu1KHDpXux2OpDXaFSeJJ0JLzm2CUL', 'sk-6H1MhvKz6w0cdwYrFAmJimwxtTGn0HZUQTiLvLULTmS5VXoB', 'sk-O17x2sgvkqEqJGYXuseuy51qnGUcrJx7pVSeYGJ3ex7Ftt9T', 'sk-51XtwSEezwPEXe4xFHc4JP7qEhPSdTIdUDTJCUQL8gpv2FjG', 'sk-vizcpmBt9qeFWiJArdOOm4N5noGMP45QA65HPdvq4qU2YqXQ', 'sk-IwfGhLfvO4lluMeqbMBwUhBbO0OywYCWfLZ8F82Ejlozg5C1', 'sk-X6a0CVBeLNaYJ5YB4dwDSrLoiM9Tt0XabJpLYEsJzjWRqH2P', 'sk-JO2tjYC3XU7bBRSQ7NnXm6IFWJt818YMjtFV7nGyMz2Uu6Ii', 'sk-P4GiiP2JVAKEZvSqGkBHi5LGYjiTjl38E6zwMWcZp8sZ6JpM', 'sk-Exv67OpDhNEfUgynaehGeP7j824zpJnYnUtBMXmsiJfD7VaE', 'sk-zA9ekSfj4zn58eqdNct4kjyVv4E1YyWoAA1UA6YF56JTmZwd', 'sk-4Pk2X9WKIQQPlJktTB0xTp2B09dLfBijCJokd8HOl0go2age']
api_key_list_pp = ['sk-AZdbOFTXh0Vlxh9fGm4HdPOTAREuG9OmpQxMjO0ux7F7Men1', 'sk-ZkjdHmFakBIPNHVbWpXRmI47WuTXFtJ62sKqMkcqsddMLTam', 'sk-M7omEeEhDSLWCw0C0leQTLH3tYC9cutAPoCzMKC8H3LpTbMq', 'sk-6n5w5fUK7Hk6FBUtqqXhr3OYFPsbhGZA49HKGb1DS5xtRd9t', 'sk-YbUR7A18w4KGEjC1wOn2oaB4D6keDJWBfOfjxZtb8HuZVidA', 'sk-UR8Tcr0HuAKNmfB4NQxdHqvbBN1nq8lEjzI4HZIdr5lpklvs', 'sk-2T7z0Xwp4cACuFXB9yVajxnEd2obnj5mj9jFRx6RGBKhfJ3R', 'sk-hyyUzYg0cNKCFhxtqUr70POXNu88MtiKia7WV6ZoX2ZxQSBG', 'sk-kmmZYVsglQtoNYHKHO64F0O8rCKZG4PREGBomFhnbSe3rfWO', 'sk-E0NV8zEu6maH9m0vM7ZOq7YpPZvcrgr6fxCpXXtIi6pt2FJi', 'sk-8P9thM3bXLoYT7aScQ0dmOroVzxxAOJ1h5ICnNiSxYFiKMbc', 'sk-UvZEyIIZSAcwnQXXwWtk32bQz3ByT0k06hO9jRjg0It4jx29', 'sk-Hrh7vCfsThVE8AiDDfTYpA3raSFkBbruqEDZTcX4hclfcy0F', 'sk-zbYoZ9Cf9b8meEqfrZypfMuMMONrdqgpbq5GZZJsiMUyVaWh', 'sk-ebROS1SewsBqkmU24MapNVni21k3Wf1p5lOVb8zjdi8SW58z', 'sk-W9Kr9Qve99w3nVD5uTgG3jU3EHA47wbeZWyi4su8jECZ51ZM', 'sk-JC3bjMnFmUDZ0zNt6F0voTWGjr4KwAiDZZ4efNIi5PFw0SBE', 'sk-Ae1ncrhxeUBvktpRIoWWHcof0jJr4bq6QfKkyMaX9aptsMa9', 'sk-GTtR9Yvvq9cQDgODDAxFVDdJe8Tg0uM0CW8xHRBKLDWopxdL', 'sk-j8TthhAGCHZcvF6heBteHMK0cBQGTfmjlQ4bEMtBNkcOjOzR']
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature", type=float, default=0.2, help=""
    )
    parser.add_argument(
        '--dataset', default='mmqa',
        help="dataset",
        choices=["mmqa", "webqa","mmcovqa" ]
    )
    parser.add_argument(
        '--KG_model', default='ComplEx',
        help="train kg model",
    )

    parser.add_argument(
        "--engine", default='gpt-3.5-turbo-16k', help="llama2-7b, llama3-8b, gpt-3.5-turbo-16k",
        choices=["llama2-7b", "llama3-8b", "gpt-3.5-turbo-16k", "gpt-4o", "deepseek-r1"]
    )
    parser.add_argument(
        "--inference_method", default='api', help=""
    )
    parser.add_argument(
        # "--api_key", default='sk-yKYFIFbf796HWSU7XdRHbz4c2lLGQgY5TRCgXRALfsSfFyvH' #gpt-4o
        "--api_key", default=api_key_list_pp[:4]
    )
    parser.add_argument(
        # "--api_base", default='https://xiaoai.plus/v1', help="api_base"
        "--api_base", default='https://www.ppapi.vip/v1', help="api_base"
    )
    parser.add_argument(
        "--test_start", default=0, type = int, help ='start index'
    )
    parser.add_argument(
        "--test_end", default=100, type = int, help ='end index'
    )
    parser.add_argument(
        '--result_path', default='/home/cmf/multiQA/KG_LLM/result/mmqa', help="your result output path"
    )
    parser.add_argument(
        '--dataset_path', default='/home/cmf/multiQA/data/MMQA/data/', help="your dataset path"
    )
    parser.add_argument(
        '--SHOW_LOGGER', default=True
    )

    parsed_args = parser.parse_args()
    return parsed_args

def train_KG():
    args = parse_arguments()
    print_exp(args)

    db_path = '/data/cmf/mmqa/kg/kg.db'

    questions = load_data(args)
    error_log_path = "/home/cmf/multiQA/KG_LLM/KGE/error.txt"
    for idx in range(811, len(questions)):
        print(f"----------{idx+1}:{questions[idx]['qid']}-------------")
        question_id = questions[idx]['qid']
        data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]

        train_path = f'/data/cmf/mmqa/KGE/{question_id}/train.txt'
        directory = os.path.dirname(train_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_save_path = f'/data/cmf/mmqa/KGE/{question_id}/trained_model.pkl'
        triples_save_path = f'/data/cmf/mmqa/KGE/{question_id}/trained_triples.pkl'
        trainer = KGTrainer(train_path, model_save_path, triples_save_path)
        try:
            trainer.run(db_path, data_id_list)
        except Exception as e:
            error_message = f"Index: {idx}, Question ID: {question_id}, Error: {str(e)}\n"
            print(f"An error occurred: {error_message}")  # 可选，打印错误信息
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(error_message)

def read_qids(file_path):
    """
    读取 slow_inference_qids.txt 文件，并返回一个包含 qid 的列表。
    
    :param file_path: 需要读取的文件路径
    :return: qid 列表
    """
    slow_qids = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            slow_qids = [line.strip() for line in f.readlines()]
        print(f"成功读取 {len(slow_qids)} 个 qid")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到！")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

    return slow_qids

def get_runed_qid(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 加载 JSON 数据
            qid_list = [item['qid'] for item in data.values()]  # 提取每个 qid
    except Exception as e:
        return []
    return qid_list


async def main():

    run_qid_list = get_runed_qid('/home/cmf/multiQA/KG_LLM/result/mmqa/gpt-3.5-turbo-16k/test.json')
    args = parse_arguments()
    print_exp(args)

    questions = load_data(args)
    KG = load_KG(args)
    
 
    dataset_path = args.dataset_path

    lancedb_path = "/data/cmf/mmqa/kg/lancedb_data"  # LanceDB 数据库存储目录
    os.makedirs(lancedb_path, exist_ok=True)  # 确保目录存在


    collections = {
        "entity": LanceDBVectorStore("entity_collection", lancedb_path),
        "relation_description":LanceDBVectorStore("relation_description_collection", lancedb_path)
    }

    for store in collections.values():
        store.connect()

    apply_type = AppliactionType.QUESTION 
    question_entity_name_embedder, question_text_embedder, question_image_embedder =load_embedder_by_appliaction(apply_type)
    apply_type = AppliactionType.GRPAH
    entity_name_embedder, text_embedder, image_embedder = load_embedder_by_appliaction(apply_type)
    entity_name_embedder, text_clip_embedder, image_embedder = load_embedder_by_appliaction(AppliactionType.CLIP)
    
    result_path =  os.path.join(args.result_path, f"{args.engine}_test_error_1_1.json")
    result = {}

    error_log_filename = os.path.join(args.result_path, "error1_1.txt")

    error_qid_file = "/home/cmf/multiQA/KG_LLM/result/mmqa/gpt-3.5-turbo-16k/mmqa_error.txt"
    error_qid_list = read_qids(error_qid_file)
    print(len(error_qid_list))
    now_run_qid_list = get_runed_qid(result_path)

    with open(error_log_filename, 'a') as error_log:
        for idx in range(0, 500):
            try:
                question = questions[idx].get("question", "")
                question_id = questions[idx].get('qid','')
                if question_id not in error_qid_list or question_id in now_run_qid_list:
                    continue
                # if question_id in run_qid_list or question_id in now_run_qid_list:
                #     continue
                question_type = questions[idx].get("question_type", "")  
                modalities = questions[idx].get("modalities", '')

                log_filename = os.path.join(args.result_path, args.engine, "log_test_error_1_1", f"{idx+1}_{question_id}.txt")
                directory = os.path.dirname(log_filename)
                os.makedirs(directory, exist_ok=True)
                logger = ThreadLogging(log_filename)
                logger.log_clean()

                retrieve_log_filename = os.path.join(args.result_path, 'retrieve', "log_test_error_1_1", f"{idx+1}_{question_id}.txt")
                directory = os.path.dirname(retrieve_log_filename)
                os.makedirs(directory, exist_ok=True)
                retrieve_logger = ThreadLogging(retrieve_log_filename)
                retrieve_logger.log_clean()

                print(f'正在处理{idx}:{question_id}')
                
                idx_question_id = f'{idx+1}_{question_id}'
                result[idx_question_id] = {}

                result[idx_question_id]['qid'] = question_id
                result[idx_question_id]['type'] = question_type
                result[idx_question_id]['modalities'] = modalities
                result[idx_question_id]['question'] = question
                result[idx_question_id]['answer'] = questions[idx].get('answer', [])
                result[idx_question_id]['supporting_context'] = questions[idx].get('supporting_context', [])
                
                if args.dataset == 'mmqa'  :
                    data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
                elif args.dataset == 'webqa':
                    data_id_list = questions[idx]['facts_list']
                else:
                    data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + questions[idx]["table_id"]

                entity_link_hash = KG.get_entity_link(idx)
                kg_entities_name = KG.get_kg_entity_names(data_id_list)

                # LLM = ModelFactory.create_model(args)
                detail_logger = ''
                llm_list = [ModelFactory.create_model(args, 0), ModelFactory.create_model(args, 1), ModelFactory.create_model(args, 2), ModelFactory.create_model(args, 3)]
                base_config = BaseConfig(llm_list, logger, detail_logger, args)
                db_path = '/data/cmf/mmqa/kg/kg.db'
                kg_config = KgConfig(KG, db_path, kg_entities_name, entity_link_hash, data_id_list)
                base_config.logger.log_info(f'question:{question}')
                
                orginal_question = question
                question = question.upper()

                query_entity_name_embedder = question_entity_name_embedder
                retrieve_config = RetrieveConfig(retrieve_logger ,base_config, kg_config, collections)
                retrieve_entities_phase = RetrieveEntitiesPhase(retrieve_config, query_entity_name_embedder)


                train_path = f'/data/cmf/mmqa/KGE/{question_id}/train.txt'
                # directory = os.path.dirname(train_path)
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                kge_flag = 1
                model_save_path = f'/data/cmf/mmqa/KGE/{question_id}/trained_model.pkl'
                triples_save_path = f'/data/cmf/mmqa/KGE/{question_id}/trained_triples.pkl'
                if not os.path.exists(model_save_path):
                    kge_flag = 0

                for store in collections.values():
                    store.filter_by_data_id(data_id_list)
                    
                question_embedding = question_text_embedder([question])[0]
                kge_retrieve_phase = KGERetrieval(args, db_path, base_config, kg_config, retrieve_config, train_path, model_save_path, triples_save_path)
                kg_reason_processor = KGReasonProcessor(base_config, kg_config, retrieve_entities_phase, kge_retrieve_phase)
                reason_phase = ReasoningPhase(base_config, kg_reason_processor)
                answer, inference_time, data_inference_time, organize_time, qa_time, get_intermediate_entity_time, validate_answer_time = await reason_phase.run(orginal_question, question, question_embedding, data_id_list, kge_flag)
                print(answer)
                result[idx_question_id][f'{args.engine}_answer'] = answer
                result[idx_question_id][f'inference_time'] = inference_time
                result[idx_question_id][f'data_inference_time'] = data_inference_time
                result[idx_question_id][f'organize_time'] = organize_time
                result[idx_question_id][f'qa_time'] = qa_time
                result[idx_question_id][f'get_intermediate_entity_time'] = get_intermediate_entity_time
                result[idx_question_id][f'validate_answer_time'] = validate_answer_time

                append_to_json_file({idx_question_id:result[idx_question_id]}, result_path)
            except Exception as e:
                error_message = f"Error processing question {question_id} (idx: {idx+1}): {str(e)}\n"
                error_log.write(error_message)
                print(f"Error processing question {question_id}: {e}")
if __name__ == "__main__":
    asyncio.run(main())
    

