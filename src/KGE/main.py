import os
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import Model
import sqlite3
import logging
import pickle  # 用于保存和加载模型
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import RobertaTokenizer, RobertaModel
import random
from data.MMQA.data_processor import MMQADataProcessor
import re
from collections import defaultdict


# 数据路径
train_path = '/home/cmf/multiQA/KG_LLM/KGE/data/train.txt'
model_save_path = '/home/cmf/multiQA/KG_LLM/KGE/data/trained_model.pkl'
triples_save_path = '/home/cmf/multiQA/KG_LLM/KGE/data/trained_triples.pkl'
db_path = '/home/cmf/multiQA/KG_LLM/vector_store/test_data/kg/kg.db'
retrieve_file_path = '/home/cmf/multiQA/KG_LLM/vector_store/retrieve_result/retrieve_by_entityName_k_10'
test_kg_path = '/home/cmf/multiQA/KG_LLM/KGE/data/kg.db'
def create_kg(test_kg_path, data_id_list):
    connection = sqlite3.connect(test_kg_path)
    cursor = connection.cursor()

    # 创建 relationships 表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        head_entity TEXT NOT NULL,
        relation TEXT NOT NULL,
        tail_entity TEXT NOT NULL
    )
    """)
    print("Table 'relationships' created successfully.")
    cursor.close()
    connection.close()

def get_kg_entities(data_id_list):
    data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
    entity_list = []
    with sqlite3.connect(test_kg_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        query = f"SELECT head_entity, tail_entity FROM relationships "
        cursor.execute(query)
        results = cursor.fetchall()
        for head_entity, tail_entity in results:
            entity_list.append(head_entity)
            entity_list.append(tail_entity)
        return list(set(entity_list))

def get_relations(data_id_list):
    data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
    relation_list = []
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type  FROM relationships where data_id IN ({data_id_list_str})"
        cursor.execute(query)
        results = cursor.fetchall()
        for data_id, head_entity, tail_entity, relation, description, text, data_type in results:
            relation_list.append((head_entity, relation, tail_entity))
        return relation_list
    
def insert_relations(relation_list):
    connection = sqlite3.connect(test_kg_path)
    cursor = connection.cursor()
    for head_entity, relation, tail_entity in relation_list:

        cursor.execute("""
        INSERT INTO relationships (head_entity, relation, tail_entity) 
        VALUES (?, ?, ?)
        """, (head_entity, relation, tail_entity))
    connection.commit()
    print(f"Inserted {len(relation_list)} relations into 'relationships' table.")
    cursor.close()
    connection.close()





def process_retrieve_file(retrieve_file_path, question_id):
    """
    处理文件，将实体名与相关实体列表映射到result字典中。
    :param file_path: 文件路径
    :return: result字典 {实体名: [相关实体列表]}
    """
    result = defaultdict(list)
    file_path = os.path.join(retrieve_file_path, f'{question_id}.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_entity = None

    # 解析每一行
    for line in lines:
        # 匹配实体类别的行，如：与 LOGO 相关的前 10 项为：
        match_entity_header = re.match(r"与\s(.+?)\s相关的前\s\d+\s项为：", line)
        if match_entity_header:
            current_entity = match_entity_header.group(1)  # 提取当前实体名称
            continue
        
        # 匹配实体条目的行，如：1. entity_id: 80492, entity_name: LOGO, ...
        match_entity_item = re.match(r"\d+\.\sentity_id:\s\d+,\sentity_name:\s(.+?),", line)
        if match_entity_item and current_entity:
            related_entity = match_entity_item.group(1)  # 提取相关实体名称
            result[current_entity].append(related_entity)
    
    return result

def get_data_id_by_entity(db_path, entity, data_id_list):
    data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
    result_data_id_list = []
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        query = f"SELECT data_id FROM entities where data_id IN ({data_id_list_str}) and entity = ?"
        cursor.execute(query, (entity,))
        results = cursor.fetchall()
        result_data_id_list = [entity[0] for entity in results]
    return result_data_id_list

def get_entities_with_hops(db_path, start_entity, hops):
    """
    查询以某个实体为中心，往外几跳的所有相关实体和关系。
    
    :param db_path: 数据库路径
    :param start_entity: 起始实体
    :param hops: 跳数
    :return: 包含所有相关实体和关系的列表
    """
    # 连接到数据库
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    # 创建一个集合用于存储已访问的实体，防止重复查询
    visited_entities = set()
    visited_entities.add(start_entity)
    
    # 用于存储最终结果的列表
    result = []
    
    # 当前层次的实体集合（初始为 start_entity）
    current_level_entities = {start_entity}
    
    for hop in range(hops):
        # 用于存储下一跳的实体
        next_level_entities = set()
        
        # 查询当前层次的每个实体的所有关系
        for entity in current_level_entities:
            cursor.execute("""
                SELECT head_entity, relation, tail_entity 
                FROM relationships 
                WHERE head_entity = ? OR tail_entity = ?
            """, (entity, entity))
            
            # 处理查询结果
            for head_entity, relation, tail_entity in cursor.fetchall():
                # 将三元组加入结果集
                result.append(head_entity)
                result.append(tail_entity)
                
                # 如果 tail_entity 是新的实体，加入下一层次
                if head_entity == entity and tail_entity not in visited_entities:
                    next_level_entities.add(tail_entity)
                    visited_entities.add(tail_entity)
                
                # 如果 head_entity 是新的实体，加入下一层次
                if tail_entity == entity and head_entity not in visited_entities:
                    next_level_entities.add(head_entity)
                    visited_entities.add(head_entity)
        
        # 更新当前层次为下一层次
        current_level_entities = next_level_entities
    
    # 关闭数据库连接
    connection.close()
    
    return result


def get_related_entity_by_entity(entity_name, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        entity_list = []
        with sqlite3.connect(test_kg_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            query = f"SELECT id, head_entity, tail_entity, relation FROM relationships WHERE (head_entity = ? or tail_entity= ?)"
            cursor.execute(query, (entity_name, entity_name))
            results = cursor.fetchall()
            for relation_id, head_entity, tail_entity, relation in results:
                entity_list.append(head_entity)
                entity_list.append(tail_entity)
        return entity_list

def write_triple_to_txt(db_path, train_path, idx):
    if not os.path.exists(train_path):
        data_processor = MMQADataProcessor()
        questions,text_questions,table_questions,image_questions = data_processor.get_dev_information()
        question = questions[idx]
        data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]

        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        relation_list = []
        with sqlite3.connect(db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type  FROM relationships where data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            for data_id, head_entity, tail_entity, relation, description, text, data_type in results:
                relation_list.append((head_entity, relation, tail_entity))

        try:
            with open(train_path, 'w', encoding='utf-8') as f:
                for head_entity, relation, tail_entity in relation_list:
                    f.write(f"{head_entity}\t{relation}\t{tail_entity}\n")
            print(f"成功将 {len(relation_list)} 个三元组写入到 {train_path}")
        except Exception as e:
            print(f"写入文件时发生错误: {e}")



def generate_complete_knowledge_graph_with_equivalences(table_data):
    """
    根据表格数据生成完整的知识图谱三元组 (head_entity, relation, tail_entity)，并处理等价实体。
    :param table_data: 包含表格数据的字典。
    :return: 知识图谱的三元组列表。
    """
    knowledge_graph = []
    equivalences = []  # 存储等价实体的三元组

    rows = table_data["table"]["table_rows"]
    columns = table_data["table"]["header"]

    # 提取列名
    column_names = [col["column_name"] for col in columns]

    # 遍历表格的每一行
    for row in rows:
        head_entity = None
        for i, cell in enumerate(row):
            # 处理头实体（假定 "Team" 列为头实体列）
            if i == 1:  # 假定第二列 "Team" 是头实体
                head_entity = cell.get("links", [{}])[0].get("wiki_title", cell["text"])
                # 如果有等价实体关系（wiki_title 和 text 不同）
                if "links" in cell and cell["links"] and cell["text"] != head_entity:
                    equivalences.append((cell["text"], "equivalent_to", head_entity))

            # 如果有头实体，则生成三元组
            if head_entity:
                relation = column_names[i]
                tail_entity = cell.get("links", [{}])[0].get("wiki_title", cell["text"])  # 如果有 wiki_title，优先使用它
                knowledge_graph.append((head_entity, relation, tail_entity))

    return knowledge_graph, equivalences


# Step 1: 数据准备
# def prepare_data():
#     """
#     准备训练集文件，如果文件不存在则创建一个示例文件。
#     """
#     if not os.path.exists(train_path):
#         with open(train_path, 'w') as f:
#             f.write("Q1\tR1\tT1\n")
#             f.write("Q2\tR2\tT2\n")
#             f.write("Q3\tR3\tT3\n")
#     print(f"训练数据保存在 {train_path}")


# Step 2: 训练模型
def train_model(model_name='Complex', embedding_dim=200, num_epochs=200, batch_size=16, learning_rate=0.01):
    """
    根据用户选择的模型训练知识图谱嵌入模型。
    """
    # 加载训练数据
    logging.basicConfig(level=logging.DEBUG)
    
    training_triples = TriplesFactory.from_path(train_path, create_inverse_triples=True)
    training_triples, testing_triples = training_triples.split([0.8, 0.2])

    # 训练模型
    pipeline_result = pipeline(
        model=model_name,
        training=training_triples,
        testing=testing_triples,
        model_kwargs={
            'embedding_dim': embedding_dim,
        },
        optimizer='Adam',
        optimizer_kwargs={
            'lr': learning_rate,
        },
        negative_sampler="basic",
        training_kwargs={
            'num_epochs': num_epochs,
            'batch_size': batch_size,
        },
        random_seed=42,
        negative_sampler_kwargs=dict(num_negs_per_pos=50)
        # device='cpu',  # 修改为 'cuda' 以使用 GPU
    )
    print(f"模型 {model_name} 训练完成！")
    return pipeline_result, training_triples


# Step 3: 保存模型
def save_model_and_triples(pipeline_result, training_triples, model_save_path, triples_save_path):
    """
    保存模型和训练三元组对象。
    :param pipeline_result: PyKEEN pipeline 训练结果
    :param training_triples: PyKEEN TriplesFactory 对象
    :param model_save_path: 模型保存路径
    :param triples_save_path: 三元组保存路径
    """
    try:
        # 保存模型
        with open(model_save_path, 'wb') as model_file:
            pickle.dump(pipeline_result.model, model_file)
        print(f"模型保存成功到 {model_save_path}")

        # 保存训练三元组
        with open(triples_save_path, 'wb') as triples_file:
            pickle.dump(training_triples, triples_file)
        print(f"训练三元组保存成功到 {triples_save_path}")

    except Exception as e:
        print(f"保存时发生错误: {e}")

# Step 4: 加载模型
def load_model_and_triples(model_save_path, triples_save_path):
    """
    加载保存的模型和训练三元组对象。
    :param model_save_path: 模型保存路径
    :param triples_save_path: 三元组保存路径
    :return: 加载的模型和训练三元组对象
    """
    try:
        # 加载模型
        with open(model_save_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"模型加载成功从 {model_save_path}")

        # 加载训练三元组
        with open(triples_save_path, 'rb') as triples_file:
            training_triples = pickle.load(triples_file)
        print(f"训练三元组加载成功从 {triples_save_path}")

        return model, training_triples

    except Exception as e:
        print(f"加载时发生错误: {e}")
        return None, None


import torch
from sklearn.metrics.pairwise import cosine_similarity

def generate_relation_embedding(input_relation, embedding_model, existing_relation_embeddings):
    """
    使用文本嵌入模型生成一个新的关系嵌入。
    """
    # 使用文本嵌入模型生成输入关系的嵌入
    input_embedding = embedding_model.get_sentence_vector(input_relation)

    # 如果需要，可以使用已有的关系嵌入进行投影或映射（例如 PCA）
    # 这里直接返回生成的嵌入
    return torch.tensor(input_embedding, dtype=torch.float32)

# Step 5: 链接预测
pretrained_weights = 'roberta-base'
tokenizer_class = RobertaTokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

def process_question(question, question_entity_list):
    for entity in question_entity_list:
        question = question.replace(entity, 'NE')
    question_tokenized, attention_mask = tokenize_question(question)
    
    return question_tokenized, attention_mask

def pad_sequence(arr, max_len=128):
    num_to_add = max_len - len(arr)
    for _ in range(num_to_add):
        arr.append('<pad>')
    return arr

def tokenize_question(question):
    question = "<s> " + question + " </s>"
    question_tokenized = tokenizer.tokenize(question)
    question_tokenized = pad_sequence(question_tokenized, 64)
    question_tokenized = torch.tensor(tokenizer.encode(question_tokenized, add_special_tokens=False))
    attention_mask = []
    for q in question_tokenized:
        if q == 1:
            attention_mask.append(0)
        else:
            attention_mask.append(1)
    return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

def getQuestionEmbedding(question_tokenized, attention_mask, device='cuda'):
    """
    使用预训练的 RoBERTa 模型生成问题的嵌入。
    :param question_tokenized: 已经 tokenized 的问题 (形状为 [batch_size, seq_length])
    :param attention_mask: Attention mask (形状为 [batch_size, seq_length])
    :return: 问题的嵌入 (形状为 [batch_size, hidden_size])
    """
    # 加载预训练的 RoBERTa 模型
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)

    # 确保输入为二维张量
    if question_tokenized.dim() == 1:
        question_tokenized = question_tokenized.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    
    question_tokenized = question_tokenized.to(device)
    attention_mask = attention_mask.to(device)

    # 获取模型的输出
    roberta_outputs = roberta_model(input_ids=question_tokenized, attention_mask=attention_mask)

    # 获取最后一层隐藏状态 (形状为 [batch_size, seq_length, hidden_size])
    roberta_last_hidden_states = roberta_outputs.last_hidden_state

    # 提取 CLS token 的嵌入 (CLS token 的位置是第一个位置，即索引 0)
    cls_embedding = roberta_last_hidden_states[:, 0, :]  # 形状为 [batch_size, hidden_size]

    return cls_embedding

# def predict_tail_entity(
#     question_entity, 
#     question_embedding, 
#     candidate_entities, 
#     model, 
#     entity_to_id, 
#     relation_to_id, 
#     embedding_model
# ):
#     """
#     根据头实体 h 和新关系 r 预测尾实体 t。
#     如果关系不存在，则动态生成一个新的关系嵌入。
#     """
#     h_id = entity_to_id[question_entity]
    
    

#     # 获取实体嵌入
#     h_embedding = model.entity_representations[0](indices=torch.tensor([h_id]))
#     t_ids = [entity_to_id[candidate] for candidate in candidate_entities]
#     t_embeddings = model.entity_representations[0](indices=torch.tensor(t_ids))

    
#     scores = -torch.norm(h_embedding + question_embedding - t_embeddings, p=2, dim=1).tolist()
#     sorted_candidates = sorted(zip(candidate_entities, scores), key=lambda x: x[1], reverse=True)
#     return sorted_candidates

def predict_tail_entity(
    question_entity, 
    question_embedding, 
    candidate_entities, 
    model, 
    entity_to_id, 
    relation_to_id, 
    model_name="TransE"
):
    """
    根据头实体 h 和新关系 r 预测尾实体 t，并支持不同模型的分数计算逻辑。
    :param question_entity: 问题中的头实体 h
    :param question_embedding: 问题的关系嵌入表示
    :param candidate_entities: 候选尾实体列表 t
    :param model: 知识图谱嵌入模型
    :param entity_to_id: 实体到 ID 的映射
    :param relation_to_id: 关系到 ID 的映射
    :param embedding_model: 用于生成嵌入的模型
    :param model_name: 模型名称，用于选择计算逻辑（如 "TransE", "ComplEx", "RotatE"）
    :return: 排序后的候选实体及其分数
    """
    print('------------')
    print(question_entity)
    h_id = entity_to_id[question_entity]

    # 获取头实体嵌入
    h_embedding = model.entity_representations[0](indices=torch.tensor([h_id]))
    t_ids = [entity_to_id[candidate] for candidate in candidate_entities]
    t_embeddings = model.entity_representations[0](indices=torch.tensor(t_ids))
    linear_projection = nn.Linear(768, 200).to('cuda')
    print(question_embedding.device)
    question_embedding = linear_projection(question_embedding)
    

    # 不同模型的得分计算逻辑
    if model_name == "TransE":
        # TransE: d(h + r, t)
        scores = -torch.norm(h_embedding + question_embedding - t_embeddings, p=2, dim=1).tolist()

    elif model_name == "ComplEx":
        
        # ComplEx: Real(<h, r, t>)
        h_re, h_im = torch.chunk(h_embedding, 2, dim=-1)
        r_re, r_im = torch.chunk(question_embedding, 2, dim=-1)
        t_re, t_im = torch.chunk(t_embeddings, 2, dim=-1)
        r_re.to('cuda:0')
        r_im.to('cuda:0')
        print(h_re.device, h_im.device, r_re.device, r_im.device, t_re.device, t_im.device)


        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_re
        complex_score = re_score * t_re + im_score * t_im
        scores = torch.sum(complex_score, dim=-1).real.tolist() 

    elif model_name == "RotatE":
        # RotatE: ||h ∘ r - t||, where ∘ is element-wise product
        phase_relation = question_embedding / (torch.norm(question_embedding, p=2, dim=-1, keepdim=True))
        h_rotated = h_embedding * torch.exp(1j * phase_relation)
        scores = -torch.norm(h_rotated - t_embeddings, p=2, dim=1).tolist()

    else:
        # 默认使用欧几里得距离计算
        print(f"Unknown model name: {model_name}. Using default Euclidean distance.")
        scores = -torch.norm(h_embedding + question_embedding - t_embeddings, p=2, dim=1).tolist()

    # 根据得分排序候选实体
    sorted_candidates = sorted(zip(candidate_entities, scores), key=lambda x: x[1], reverse=True)
    return sorted_candidates

def predict_head_entity(
    question_tail, 
    question_embedding, 
    candidate_entities, 
    model, 
    entity_to_id, 
    relation_to_id, 
    model_name="TransE"
):
    """
    根据关系 r 和尾实体 t 预测头实体 h，并支持不同模型的分数计算逻辑。
    :param question_embedding: 问题中的关系 r
    :param question_tail: 问题中的尾实体 t
    :param candidate_entities: 候选头实体列表 h
    :param model: 知识图谱嵌入模型
    :param entity_to_id: 实体到 ID 的映射
    :param relation_to_id: 关系到 ID 的映射
    :param embedding_model: 用于生成嵌入的模型
    :param model_name: 模型名称，用于选择计算逻辑（如 "TransE", "ComplEx", "RotatE"）
    :return: 排序后的候选实体及其分数
    """
    # 获取尾实体和关系的 ID
    t_id = entity_to_id[question_tail]

    # 获取尾实体嵌入
    t_embedding = model.entity_representations[0](indices=torch.tensor([t_id]))
    h_ids = [entity_to_id[candidate] for candidate in candidate_entities]
    h_embeddings = model.entity_representations[0](indices=torch.tensor(h_ids))
    linear_projection = nn.Linear(768, 200).to('cuda')
    question_embedding = linear_projection(question_embedding)

    # 不同模型的得分计算逻辑
    if model_name == "TransE":
        # TransE: d(h + r, t)
        scores = -torch.norm(h_embeddings + question_embedding - t_embedding, p=2, dim=1).tolist()

    elif model_name == "ComplEx":
        # ComplEx: Real(<h, r, t>)
        h_re, h_im = torch.chunk(h_embeddings, 2, dim=-1)
        r_re, r_im = torch.chunk(question_embedding, 2, dim=-1)
        t_re, t_im = torch.chunk(t_embedding, 2, dim=-1)
        print(h_re.device, h_im.device, r_re.device, r_im.device)


        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_re
        complex_score = re_score * t_re + im_score * t_im
        scores = torch.sum(complex_score, dim=-1).real.tolist() 

    elif model_name == "RotatE":
        # RotatE: ||h ∘ r - t||, where ∘ is element-wise product
        phase_relation = question_embedding / (torch.norm(question_embedding, p=2, dim=-1, keepdim=True))
        h_rotated = h_embeddings * torch.exp(1j * phase_relation)
        scores = -torch.norm(h_rotated - t_embedding, p=2, dim=1).tolist()

    else:
        # 默认使用欧几里得距离计算
        print(f"Unknown model name: {model_name}. Using default Euclidean distance.")
        scores = -torch.norm(h_embeddings + question_embedding - t_embedding, p=2, dim=1).tolist()

    # 根据得分排序候选实体
    sorted_candidates = sorted(zip(candidate_entities, scores), key=lambda x: x[1], reverse=True)
    return sorted_candidates


# def predict_head_entity(question_relation, question_tail, candidate_entities, model, entity_to_id, relation_to_id):
#     """
#     根据关系 r 和尾实体 t 预测头实体 h。
#     """
#     r_id = relation_to_id[question_relation]
#     t_id = entity_to_id[question_tail]
#     h_ids = [entity_to_id[candidate] for candidate in candidate_entities]
    
#     scores = model.score_h(
#         r=torch.tensor([r_id]),
#         t=torch.tensor([t_id]),
#         candidates=torch.tensor(h_ids)
#     ).tolist()
#     sorted_candidates = sorted(zip(candidate_entities, scores), key=lambda x: x[1], reverse=True)
#     return sorted_candidates


# 主程序
def main():
    # Step 1: 数据准备
    # write_triple_to_txt(db_path, train_path, 99)

    # # 用户选择模型
    # available_models = ['TransE', 'DistMult', 'ComplEx', 'RotatE']
    # selected_model = 'ComplEx'
    # # print(f"可用模型: {available_models}")
    # # selected_model = input(f"请选择模型 (默认: TransE): ").strip() or 'TransE'
    
    # # if selected_model not in available_models:
    # #     print(f"无效的模型名称: {selected_model}，将使用默认模型 TransE。")
    # #     selected_model = 'TransE'

    # # Step 2: 训练模型
    # pipeline_result, training_triples = train_model(model_name=selected_model)

    # # 保存训练好的模型
    # save_model_and_triples(pipeline_result, training_triples, model_save_path, triples_save_path)
    # 加载模型
    model, training_triples = load_model_and_triples(model_save_path, triples_save_path)

    # 获取实体和关系的映射
    entity_to_id = training_triples.entity_to_id
    relation_to_id = training_triples.relation_to_id

    # Step 3: 链接预测
    # print("\n=== 链接预测 ===")

    data_processor = MMQADataProcessor()
    questions,text_questions,table_questions,image_questions = data_processor.get_dev_information()
    idx = 99
    question = questions[idx]
    data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
    
    # create_kg(test_kg_path, data_id_list)
    # relation_list = get_relations(data_id_list)
    # insert_relations(relation_list)
    

    question = 'WHO SCORED HIGHER AL-HILAL OR AL-AHLI, DURING THE 1998-99 SAUDI PREMIER LEAGE PLAY?'
    question_id  = 'bdde9feefbf9c4533b9303cbf0260096'
    question_entity_list = ['1998-99 SAUDI PREMIER LEAGUE PLAY', '1998-99 SAUDI PREMIER LEAGUE', 'AL-HILAL', 'AL-AHLI']
    question_tokenized, attention_mask = process_question(question, question_entity_list)
    question_embedding = getQuestionEmbedding(question_tokenized, attention_mask)
    
    result = process_retrieve_file(retrieve_file_path, question_id)
    print(len(entity_to_id))
    result_entities = []
    for entity in question_entity_list:
        print(entity)
        candidate_entities = []
        related_entity = result[entity]
        
        for e in related_entity:
            print(f'与{entity}有关的实体：{e}')
            # candidate_entities = get_entities_with_hops(test_kg_path, e, hops=3)
            candidate_entities = get_kg_entities(data_id_list)
            candidate_entities = list(set(candidate_entities))
            print(f'与{e}相联的实体：{candidate_entities}')
            if not candidate_entities:
                continue
            tail_entity_list = predict_tail_entity(e, question_embedding,candidate_entities, model, entity_to_id, relation_to_id,  model_name='ComplEx')
            head_entity_list = predict_head_entity(e, question_embedding, candidate_entities, model, entity_to_id, relation_to_id,  model_name='ComplEx')
            
            for tail_entity, score in tail_entity_list:
                if score > 0:
                    result_entities.append(tail_entity)
                print(f"Entity: {tail_entity}, Score: {score}")
            for head_entity, score in head_entity_list:
                if score > 0:
                    result_entities.append(head_entity)
                print(f"Entity: {head_entity}, Score: {score}")
    result_data_id_list = []
    db_path = '/home/cmf/multiQA/KG_LLM/vector_store/test_data/kg/kg.db'
    for entity in result_entities:
        data_id = get_data_id_by_entity(db_path, entity, data_id_list)
        result_data_id_list += data_id
    print(list(set(result_data_id_list)))

    # question_relation = "R1"
    # candidate_entities_tail = ["T1", "T2", "T3"]
    # predictions_tail = predict_tail_entity(question_entity, question_relation, candidate_entities_tail, model, entity_to_id, relation_to_id)
    # print("\n尾实体预测 (h, r -> t):")
    # for entity, score in predictions_tail:
    #     print(f"Entity: {entity}, Score: {score}")

    # # 示例问题：预测头实体
    # question_tail = "T1"
    # candidate_entities_head = ["Q1", "Q2", "Q3"]
    # predictions_head = predict_head_entity(question_relation, question_tail, candidate_entities_head, model, entity_to_id, relation_to_id)
    # print("\n头实体预测 (r, t -> h):")
    # for entity, score in predictions_head:
    #     print(f"Entity: {entity}, Score: {score}")


# 运行主程序
if __name__ == "__main__":
    main()
