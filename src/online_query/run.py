import json
import logging
import re
import os
import argparse
from config.config import IMAGE_PATH, MMQA_DB
from phase.kg_construct import *
from utils.utils import print_exp, save_result, write_json, append_to_json_file
from utils.log import ThreadLogging
from data.preprocess.load_data import load_data, load_data_new, load_KG
from models.llama2_predict import predict, model_init
from models.LLM.ModelFactory import ModelFactory
from phase.get_embedding import load_dpr_model, get_text_embedding,load_index_and_mapping,find_most_similar_in_candidates,load_model_and_tokenizer,get_embeddings
from sentence_transformers import SentenceTransformer
from KG_LLM.reasoning import ReasoningPhase
from vector_score.lancedb import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_token_length", type=int, default=4096,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--example_num", type=int, default=0,
        help=" we use the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=2.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help=""
    )
    parser.add_argument(
        '--dataset', default='mmqa',
        help="dataset",
        choices=["mmqa", "webqa","mmcovqa" ]
    )
    # parser.add_argument(
    #     "--type_list_file", default="./src/format/entity_type_list.txt", type=str, help='file path'
    # )
    parser.add_argument(
        "--engine", default='gpt-3.5-turbo', help="llama2-7b, llama3-8b, gpt-3.5-turbo",
        choices=["llama2-7b", "llama3-8b", "gpt-3.5-turbo"]
    )
    parser.add_argument(
        "--model_path", default='/Llama-2-7b-chat-hf', help="your local model path"
    )
    parser.add_argument(
        "--api_key", default='sk-llqpW9khKdoFanaN1e992c3077Bd4a05A9D581BcA382F889', help="your api_key"
    )
    parser.add_argument(
        "--api_base", default='https://api.xiaoai.plus/v1', help="api_base"
    )
    parser.add_argument(
        "--test_start", default=46, type = int, help ='start index'
    )
    parser.add_argument(
        "--test_end", default=100, type = int, help ='end index'
    )
    parser.add_argument(
        "--implicit_rel", default=False, type=bool, help="implicit_relation infer"
    )
    parser.add_argument(
        "--SC", default=False, type=bool, help="self consistency"
    )
    parser.add_argument(
        '--result_path', default='result', help="your result output path"
    )
    parser.add_argument(
        '--dataset_path', default='data/', help="your dataset path"
    )
    parser.add_argument(
        '--embedding_model_path', default='/models/text_embedding/model/', help="your embedding model path"
    )
    parser.add_argument(
        '--embedding_model_name', default='sentence-transformers/all-MiniLM-L6-v2', help="your get embedding model"
    )
    parser.add_argument(
        '--index_type', default='hnsw', help="your index"
    )
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = parse_arguments()
    print_exp(args)

    questions, text_data, image_data, table_data = load_data(args)
 
    dataset_path = args.dataset_path

    index_type = args.index_type
    embedding_model_name = args.embedding_model_name
    embed_model, embed_tokenizer = load_model_and_tokenizer(embedding_model_name)
    label = embedding_model_name.replace('/', '-')
    embedding_path = os.path.join(dataset_path, 'embedding', 'kg_llm', label)
    entity_index, entity_id_to_key = load_index_and_mapping(os.path.join(embedding_path,f'entity_index_{index_type}.faiss'), os.path.join(embedding_path,f'entity_id_to_key_{index_type}.pkl'))
    entity_desp_index, entity_desp_id_to_key = load_index_and_mapping(os.path.join(embedding_path,f'entity_desp_index_{index_type}.faiss'), os.path.join(embedding_path,f'entity_desp_id_to_key_{index_type}.pkl'))
    triple_index, triple_id_to_key = load_index_and_mapping(os.path.join(embedding_path,f'triple_index_{index_type}.faiss'), os.path.join(embedding_path,f'triple_id_to_key_{index_type}.pkl'))
    triple_desp_index, triple_desp_id_to_key = load_index_and_mapping(os.path.join(embedding_path,f'triple_desp_index_{index_type}.faiss'), os.path.join(embedding_path,f'triple_desp_id_to_key_{index_type}.pkl'))
    
    db_path = ""  # LanceDB 数据库存储目录
    os.makedirs(db_path, exist_ok=True)  # 确保目录存在


    collections = {
        "entity": LanceDBVectorStore("entity_collection", db_path),
        "description": LanceDBVectorStore("description_collection", db_path),
        "image": LanceDBVectorStore("image_collection", db_path),
        "whole_image": LanceDBVectorStore("whole_image_collection", db_path),
        "image_description": LanceDBVectorStore("image_description_collection", db_path),
        "clip_description":LanceDBVectorStore("clip_description_collection", db_path),
        "relation_description":LanceDBVectorStore("relation_description_collection", db_path)
    }

    for store in collections.values():
        store.connect()

    apply_type = AppliactionType.QUESTION 
    question_entity_name_embedder, question_text_embedder, question_image_embedder =load_embedder_by_appliaction(apply_type)
    apply_type = AppliactionType.GRPAH
    entity_name_embedder, text_embedder, image_embedder = load_embedder_by_appliaction(apply_type)
    entity_name_embedder, text_clip_embedder, image_embedder = load_embedder_by_appliaction(AppliactionType.CLIP)

    kg_db_path = 'kg.db'
    
    result_path =  os.path.join(args.result_path, f"result_{args.engine}_{label}_3.json")
    result = {}



    
    for idx in range(args.test_start, args.test_end):
        print(f'正在处理{idx}')
        model, retrieveLLM, organizeJudgeLLM, FurtherReasonLLM, JudgeAnswerLLM = ModelFactory.create_model(args), ModelFactory.create_model(args), ModelFactory.create_model(args), ModelFactory.create_model(args), ModelFactory.create_model(args)
        question = questions[idx]["question"].upper()
        question_id = questions[idx]['qid']
        
        result[question_id] = {}

        result[question_id]['qid'] = question_id
        result[question_id]['question'] = question
        result[question_id]['answer'] = questions[idx]["answer"]
        result[question_id]['supporting_context'] = questions[idx]['supporting_context']
        
        data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
        KG = load_KG(args)
        kg = KG.create_kg_view(data_id_list)
        entity_links_hash = KG.get_entity_link(idx)
   
        for store in collections.values():
            store.filter_by_data_id(data_id_list)
        data_list = get_data_items_by_data_id(kg_db_path, data_id_list)
        

        save_path =  os.path.join(args.result_path, args.engine, "answer_3",  f"{label}", f"{idx+1}_{question_id}.txt")
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)

        log_filename = os.path.join(args.result_path, args.engine, "log_3", f"{label}",  f"{idx+1}_{question_id}.txt")
        directory = os.path.dirname(log_filename)
        os.makedirs(directory, exist_ok=True)

        retrieve_log_filename = os.path.join(args.result_path, args.engine, "retrieve_log_2", f"{label}",  f"{idx+1}_{question_id}.txt")
        directory = os.path.dirname(retrieve_log_filename)
        os.makedirs(directory, exist_ok=True)
        
        logger = ThreadLogging(log_filename)
        logger.log_clean()

        retrieve_logger = ThreadLogging(retrieve_log_filename)
        retrieve_logger.log_clean()

        logger.log_info(f'------正在处理第{idx+1}个问题------')
        question = question.upper()
        question_embedding = get_embeddings(question, embedding_model_name, embed_model, embed_tokenizer)
        kg_entities_name = KG.get_kg_entity_names()
        
        reasoning_phase = ReasoningPhase(model, retrieveLLM, organizeJudgeLLM, FurtherReasonLLM, JudgeAnswerLLM, KG, idx, args, logger, retrieve_logger, embed_model, embed_tokenizer, entity_index, entity_id_to_key, entity_desp_index, entity_desp_id_to_key, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_entity_k=3, top_triple_k=3)
        
        answer = reasoning_phase.run(question, question_embedding, kg_entities_name, MMQA_DB, IMAGE_PATH, entity_links_hash)
        
        result[question_id][f'{args.engine}_answer'] = answer
        append_to_json_file({question_id:result[question_id]}, result_path)

  
    