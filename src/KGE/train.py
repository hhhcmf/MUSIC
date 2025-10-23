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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature", type=float, default=0.2, help=""
    )
    parser.add_argument(
        '--dataset', default='manymodalqa',
        help="dataset",
        choices=["mmqa", "webqa","mmcovqa","manymodalqa" ]
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
        "--api_key", default=['sk-7iLRK4yyifcuG4IOeSDVKq42zuBGpY6zdqmOekPk31GRooID', 'sk-DzGXR5X1bwDvrzI2LqOZWioDHMaAZnttw3eVfusgvTF7aUC6', 'sk-SeFLp8dzqwjDLarrHqWGD1PtIDlkm4nwFS1XUtDG7zBjD1Jp', 'sk-7zII91lHFC596qgrOePxERQF6Dk8oZDs1kyVQ3j8JzNsmkyG'], help="sk-Jzv00GxY7OH7smQkiY3eg3RW3d8xWya9zKhJbXXc7cziUPIN, sk-Jzv00GxY7OH7smQkiY3eg3RW3d8xWya9zKhJbXXc7cziUPIN"
    )
    parser.add_argument(
        "--api_base", default='https://xiaoai.plus/v1', help="api_base"
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

    parsed_args = parser.parse_args()
    return parsed_args

# def train_KG():
#     args = parse_arguments()
#     print_exp(args)

#     db_path = '/data/cmf/webqa/kg/kg.db'

#     questions, image_data, text_data, text_id_list, image_id_list = load_data(args)
#     error_log_path = "/home/cmf/multiQA/KG_LLM/KGE/webqa_error.txt"
#     for idx in range(4759, len(questions)):
#         print(f"----------{idx+1}:{questions[idx]['id']}-------------")
#         question_id = questions[idx]['id']
#         data_id_list = questions[idx]["facts_list"] 

#         train_path = f'/data/cmf/webqa/KGE/{question_id}/train.txt'
#         directory = os.path.dirname(train_path)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         model_save_path = f'/data/cmf/webqa/KGE/{question_id}/trained_model.pkl'
#         triples_save_path = f'/data/cmf/webqa/KGE/{question_id}/trained_triples.pkl'
#         trainer = KGTrainer(train_path, model_save_path, triples_save_path)
#         try:
#             trainer.run(db_path, data_id_list)
#         except Exception as e:
#             error_message = f"Index: {idx}, Question ID: {question_id}, Error: {str(e)}\n"
#             print(f"An error occurred: {error_message}")  # 可选，打印错误信息
#             with open(error_log_path, "a", encoding="utf-8") as error_file:
#                 error_file.write(error_message)

def train_KG():
    args = parse_arguments()
    print_exp(args)

    
    questions= load_data(args)
    error_log_path = f"/home/cmf/multiQA/KG_LLM/KGE/{args.dataset}_error.txt"
    for idx in range(673, len(questions)):
        print(f"----------{idx+1}:{questions[idx]['qid']}-------------")
        question_id = questions[idx]['qid']
        if args.dataset == 'webqa':
            data_id_list = questions[idx]["facts_list"] 
        elif args.dataset == 'manymodalqa':
            data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + questions[idx]["table_id"]
        else:
            data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
        # print(data_id_list)
        train_path = f'/data2/cmf/{args.dataset}/KGE/{question_id}/train.txt'
        directory = os.path.dirname(train_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_save_path = f'/data2/cmf/{args.dataset}/KGE/{question_id}/trained_model.pkl'
        triples_save_path = f'/data2/cmf/{args.dataset}/KGE/{question_id}/trained_triples.pkl'
        trainer = KGTrainer(train_path, model_save_path, triples_save_path)
        try:
            db_path = f'/data2/cmf/{args.dataset}/kg/candidate_kg/{question_id}.db'
            trainer.run(db_path, data_id_list)
        except Exception as e:
            error_message = f"Index: {idx}, Question ID: {question_id}, Error: {str(e)}\n"
            print(f"An error occurred: {error_message}")  # 可选，打印错误信息
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(error_message)

if __name__ == "__main__":
    train_KG()