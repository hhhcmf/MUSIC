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
import concurrent.futures
import re
from collections import defaultdict
import argparse
from phase.new_phase.kg_construct.kg import KnowledgeGraph

# ========================= KG Database Handler =========================
class KGDatabaseHandler:
    def __init__(self, db_path, kg_config):
        self.db_path = db_path
        self.kg_config = kg_config

    def create_kg(self):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        drop_temp_tables_sql = 'DROP TABLE IF EXISTS kge_relationships;'
        cursor.execute(drop_temp_tables_sql)

        # 创建 relationships 表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS kge_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            head_entity TEXT NOT NULL,
            relation TEXT NOT NULL,
            tail_entity TEXT NOT NULL
        )
        """)
        print("Table 'kge_relationships' created successfully.")
        cursor.close()
        connection.close()

    def get_kg_entities(self):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in self.kg_config.data_id_list])
        entity_list = []
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            query = f"SELECT head_entity, tail_entity FROM relationships  where data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            for head_entity, tail_entity in results:
                entity_list.append(head_entity)
                entity_list.append(tail_entity)
            return list(set(entity_list))
    
    def get_relations(self):
        return self.kg_config.KG.get_triples(self.kg_config.data_id_list)

    def insert_relations(self, relation_list):
        connection = sqlite3.connect(self.db_path)
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

    def get_entities_with_hops(self, start_entity, hops):
        """
        查询以某个实体为中心，往外几跳的所有相关实体和关系。
        
        :param db_path: 数据库路径
        :param start_entity: 起始实体
        :param hops: 跳数
        :return: 包含所有相关实体和关系的列表
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        visited_entities = set()
        visited_entities.add(start_entity)
        result = []
        current_level_entities = {start_entity}

        for hop in range(hops):
            next_level_entities = set()
            for entity in current_level_entities:
                cursor.execute("""
                    SELECT head_entity, relation, tail_entity 
                    FROM relationships 
                    WHERE head_entity = ? OR tail_entity = ?
                """, (entity, entity))
                for head_entity, relation, tail_entity in cursor.fetchall():
                    result.append((head_entity, relation, tail_entity))
                    if head_entity == entity and tail_entity not in visited_entities:
                        next_level_entities.add(tail_entity)
                        visited_entities.add(tail_entity)
                    if tail_entity == entity and head_entity not in visited_entities:
                        next_level_entities.add(head_entity)
                        visited_entities.add(head_entity)
            current_level_entities = next_level_entities
        connection.close()
        return result

    def get_related_entity_by_entity(self, entity_name):
        entity_name_list = self.kg_config.KG.get_related_entity_name_by_entity(entity_name, self.kg_config.data_id_list)
        return entity_name_list
    
    def get_data_id_by_entity(self, entity):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in self.kg_config.data_id_list])
        result_data_id_list = []
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            query = f"SELECT data_id FROM entities where data_id IN ({data_id_list_str}) and entity = ?"
            cursor.execute(query, (entity,))
            results = cursor.fetchall()
            result_data_id_list = [entity[0] for entity in results]
        return result_data_id_list
    

# ========================= MMQA Question Processor =========================
class MMQAQuestionProcessor:
    def __init__(self, device='cuda:0'):
        self.device = device
        pretrained_weights = '/home/cmf/multiQA/loaded_models/roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        self.roberta_model = RobertaModel.from_pretrained(pretrained_weights).to(self.device)

    def process_question(self, question, question_entity_list):
        for entity in question_entity_list:
            question = question.replace(entity, 'NE')
        return self.tokenize_question(question)

    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

    def get_question_embedding(self, question_tokenized, attention_mask, device='cuda:0'):
        """
        使用预训练的 RoBERTa 模型生成问题的嵌入。
        :param question_tokenized: 已经 tokenized 的问题 (形状为 [batch_size, seq_length])
        :param attention_mask: Attention mask (形状为 [batch_size, seq_length])
        :return: 问题的嵌入 (形状为 [batch_size, hidden_size])
        """

        # 确保输入为二维张量
        if question_tokenized.dim() == 1:
            question_tokenized = question_tokenized.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        question_tokenized = question_tokenized.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 获取模型的输出
        roberta_outputs = self.roberta_model(input_ids=question_tokenized, attention_mask=attention_mask)

        # 获取最后一层隐藏状态 (形状为 [batch_size, seq_length, hidden_size])
        roberta_last_hidden_states = roberta_outputs.last_hidden_state

        # 提取 CLS token 的嵌入 (CLS token 的位置是第一个位置，即索引 0)
        cls_embedding = roberta_last_hidden_states[:, 0, :]  # 形状为 [batch_size, hidden_size]

        return cls_embedding


# ========================= KG Trainer =========================
class KGTrainer:
    def __init__(self, train_path, model_save_path, triples_save_path, device='cuda:0'):
        self.train_path = train_path
        self.model_save_path = model_save_path
        self.triples_save_path = triples_save_path
        self.device = device

    def run(self, db_path, data_id_list=[]):
        # print('1.生成训练数据')
        self.write_triple_to_txt(db_path, data_id_list)
        # self.write_all_triples(db_path)

        # Step 2: 训练模型
        # print('2.模型训练')
        pipeline_result, training_triples = self.train_model()

        # 保存训练好的模型
        self.save_model_and_triples(pipeline_result, training_triples)
    
    def write_all_triples(self, db_path):
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 查询数据（如果数据量大，使用 fetchmany 批量读取）
        cursor.execute(f"SELECT head_entity, tail_entity, relation FROM relationships")     

        # 打开文件流写入
        with open(self.train_path, "w", encoding="utf-8") as f:
            while True:
                rows = cursor.fetchmany(10000)  # 每次读取 10000 条
                if not rows:
                    break  # 读取完毕则退出
                for head_entity, tail_entity, relation in rows:
                    f.write(f"{head_entity}\t\t{relation}\t\t{tail_entity}\n")
        conn.close()
        print(f"数据已成功导出到 {self.train_path}")

    def write_triple_to_txt(self, db_path, data_id_list):#生成训练数据
        if not os.path.exists(self.train_path):
            # data_processor = MMQADataProcessor()
            # questions,text_questions,table_questions,image_questions = data_processor.get_dev_information()
            # question = questions[idx]
            # data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]

            data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
            relation_list = []
            with sqlite3.connect(db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type  FROM relationships where data_id IN ({data_id_list_str})"
                cursor.execute(query)
                # print(query)
                results = cursor.fetchall()
                for data_id, head_entity, tail_entity, relation, description, text, data_type in results:
                    if not tail_entity or not head_entity or not relation:
                        continue
                        print(data_id, head_entity, tail_entity, relation)
                    relation_list.append((head_entity.upper(), relation.upper(), tail_entity.upper()))

            try:
                with open(self.train_path, 'w', encoding='utf-8') as f:
                    for head_entity, relation, tail_entity in relation_list:
                        f.write(f"{head_entity}\t\t{relation}\t\t{tail_entity}\n")
                print(f"成功将 {len(relation_list)} 个三元组写入到 {self.train_path}")
            except Exception as e:
                print(f"写入文件时发生错误: {e}")
    def generate_complete_knowledge_graph_with_equivalences(table_data):#将具有等价的数据进行链接
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
                
    def train_model(self, model_name='Complex', embedding_dim=200, num_epochs=100, batch_size=16, learning_rate=0.01):
        """
        根据用户选择的模型训练知识图谱嵌入模型。
        """
        logging.basicConfig(level=logging.DEBUG)
        training_triples = TriplesFactory.from_path(self.train_path, create_inverse_triples=True)
        try:
            training_triples, testing_triples = training_triples.split([0.8, 0.2])
        except Exception as e:
            training_triples, testing_triples = training_triples.split([0.9, 0.1])
        

        pipeline_result = pipeline(
            model=model_name,
            training=training_triples,
            testing=testing_triples,
            device = self.device,
            model_kwargs={
                'embedding_dim': embedding_dim,
            },
            optimizer='Adam',
            optimizer_kwargs={
                'lr': learning_rate,
            },
            training_kwargs={
                'num_epochs': num_epochs,
                'batch_size': batch_size,
            },
            random_seed=42
        )
        print(f"Model {model_name} training completed!")
        return pipeline_result, training_triples

    def save_model_and_triples(self, pipeline_result, training_triples):
        """
        保存模型和训练三元组对象。
        :param pipeline_result: PyKEEN pipeline 训练结果
        :param training_triples: PyKEEN TriplesFactory 对象
        :param model_save_path: 模型保存路径
        :param triples_save_path: 三元组保存路径
        """
        with open(self.model_save_path, 'wb') as model_file:
            pickle.dump(pipeline_result.model, model_file)
        with open(self.triples_save_path, 'wb') as triples_file:
            pickle.dump(training_triples, triples_file)
        print("Model and triples saved successfully.")

    def load_model_and_triples(self):
        """
        加载保存的模型和训练三元组对象。
        :param model_save_path: 模型保存路径
        :param triples_save_path: 三元组保存路径
        :return: 加载的模型和训练三元组对象
        """
        with open(self.model_save_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(self.triples_save_path, 'rb') as triples_file:
            training_triples = pickle.load(triples_file)
        print("Model and triples loaded successfully.")
        return model, training_triples


# ========================= KG Link Predictor =========================
class KGLinkPredictor:
    def __init__(self, model, entity_to_id, relation_to_id, device = 'cuda:0'):
        self.model = model
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.device = device
    # def predict_tail_entity(self, question_entity, question_embedding, candidate_entities, model_name="TransE"):
    #     h_id = self.entity_to_id[question_entity]
    #     h_embedding = self.model.entity_representations[0](indices=torch.tensor([h_id]))
    #     t_ids = [self.entity_to_id[candidate] for candidate in candidate_entities]
    #     t_embeddings = self.model.entity_representations[0](indices=torch.tensor(t_ids))
        
    #     if model_name == "TransE":
    #         scores = -torch.norm(h_embedding + question_embedding - t_embeddings, p=2, dim=1).tolist()
    #     else:
    #         scores = -torch.norm(h_embedding - t_embeddings, p=2, dim=1).tolist()

    #     sorted_candidates = sorted(zip(candidate_entities, scores), key=lambda x: x[1], reverse=True)
    #     return sorted_candidates
    
    def predict_tail_entity(
        self,
        question_entity, 
        question_embedding, 
        candidate_entities, 
        model_name="TransE",
        top_m =2
    ):
        """
        根据头实体 h 和新关系 r 预测尾实体 t，并支持不同模型的分数计算逻辑。
        :param question_entity: 问题中的头实体 h
        :param question_embedding: 问题的关系嵌入表示
        :param candidate_entities: 候选尾实体列表 t
        :param model: 知识图谱嵌入模型
        :param embedding_model: 用于生成嵌入的模型
        :param model_name: 模型名称，用于选择计算逻辑（如 "TransE", "ComplEx", "RotatE"）
        :return: 排序后的候选实体及其分数
        """
        # print('------------')
        # print(question_entity)
        h_id = self.entity_to_id[question_entity]

        # 获取头实体嵌入
        h_embedding = self.model.entity_representations[0](indices=torch.tensor([h_id]).to(self.device))
        t_ids = []
        for candidate in candidate_entities:
            if candidate in self.entity_to_id.keys():
                t_ids.append(self.entity_to_id[candidate])
        t_embeddings = self.model.entity_representations[0](indices=torch.tensor(t_ids).to(self.device))
        linear_projection = nn.Linear(768, 200).to(self.device)
        # print(question_embedding.device)
        question_embedding = linear_projection(question_embedding)
        h_embedding = h_embedding.to(self.device)
        question_embedding = question_embedding.to(self.device)
        t_embeddings = t_embeddings.to(self.device)
        

        # 不同模型的得分计算逻辑
        if model_name == "TransE":
            # TransE: d(h + r, t)
            scores = -torch.norm(h_embedding + question_embedding - t_embeddings, p=2, dim=1).tolist()

        elif model_name == "ComplEx":
            
            
            # ComplEx: Real(<h, r, t>)
            h_re, h_im = torch.chunk(h_embedding, 2, dim=-1)
            r_re, r_im = torch.chunk(question_embedding, 2, dim=-1)
            t_re, t_im = torch.chunk(t_embeddings, 2, dim=-1)
            # r_re.to('cuda:0')
            # r_im.to('cuda:0')
            # print(h_re.device, h_im.device, r_re.device, r_im.device, t_re.device, t_im.device)


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
        return sorted_candidates[:top_m]

    def predict_head_entity(
        self,
        question_tail, 
        question_embedding, 
        candidate_entities,  
        model_name="TransE",
        top_m =2
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
        t_id = self.entity_to_id[question_tail]

        # 获取尾实体嵌入
        t_embedding = self.model.entity_representations[0](indices=torch.tensor([t_id]).to(self.device))
        h_ids = []
        for candidate in candidate_entities:
            if candidate in self.entity_to_id.keys():
                h_ids.append(self.entity_to_id[candidate])
        h_embeddings = self.model.entity_representations[0](indices=torch.tensor(h_ids).to(self.device))
        linear_projection = nn.Linear(768, 200).to(self.device)
        question_embedding = linear_projection(question_embedding)
        h_embeddings = h_embeddings.to(self.device)
        question_embedding = question_embedding.to(self.device)
        t_embedding = t_embedding.to(self.device)

        # 不同模型的得分计算逻辑
        if model_name == "TransE":
            # TransE: d(h + r, t)
            scores = -torch.norm(h_embeddings + question_embedding - t_embedding, p=2, dim=1).tolist()

        elif model_name == "ComplEx":
            # ComplEx: Real(<h, r, t>)
            h_re, h_im = torch.chunk(h_embeddings, 2, dim=-1)
            r_re, r_im = torch.chunk(question_embedding, 2, dim=-1)
            t_re, t_im = torch.chunk(t_embedding, 2, dim=-1)
            # print(h_re.device, h_im.device, r_re.device, r_im.device)


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
        return sorted_candidates[:top_m]



# ========================= Main Program =========================
class KGERetrieval:
    def __init__(self, args, db_path, base_config, kg_config, retrieve_config, train_path, model_save_path, triples_save_path):
        self.kg_handler = KGDatabaseHandler(db_path, kg_config)
        self.question_processor = MMQAQuestionProcessor()
        self.kg_trainer = KGTrainer(train_path, model_save_path, triples_save_path)
        self.args = args
        self.db_path = db_path
        self.kg_config = kg_config
        self.base_config = base_config
        self.retrieve_config = retrieve_config
    
    def process_retrieve_file(self, retrieve_file_path, question_id):
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
    def predict_tail_entities(self, e, question_embedding, candidate_entities, kg_predictor):
        return kg_predictor.predict_tail_entity(e, question_embedding, candidate_entities, model_name='ComplEx', top_m=1)

    def predict_head_entities(self, e, question_embedding, candidate_entities, kg_predictor):
        return kg_predictor.predict_head_entity(e, question_embedding, candidate_entities, model_name='ComplEx', top_m=1)

    
   
    def run(self, question, question_entity_list, data_id_list, retrieved_entity_data_list):
        self.retrieve_config.retrieve_logger.log_info("------------基于KGE进行检索----------------\n")
        retrieved_entity_data_str = ''
        for retrieved_entity_data in retrieved_entity_data_list:
            retrieved_entity_data_str += f'{retrieved_entity_data.entity_name}:{retrieved_entity_data.data_id}\n'
        self.retrieve_config.retrieve_logger.log_info(f"1.基于fasttext检索到的实体:\n{retrieved_entity_data_str}\n")
        # Example workflow
        # if self.args.dataset.lower() == 'mmqa':
        #     question = questions[idx]
        #     data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]


        # Step 1: 数据准备
        # self.kg_trainer.write_triple_to_txt(self.db_path, data_id_list)

        # 用户选择模型
        # available_models = ['TransE', 'DistMult', 'ComplEx', 'RotatE']
        # selected_model = 'ComplEx'

        # Step 2: 训练模型
        # pipeline_result, training_triples = self.kg_trainer.train_model(model_name=self.args.KG_model)

        # 保存训练好的模型
        # self.kg_trainer.save_model_and_triples(pipeline_result, training_triples)

        model, training_triples = self.kg_trainer.load_model_and_triples()
        entity_to_id = training_triples.entity_to_id
        relation_to_id = training_triples.relation_to_id

        # question = "WHO SCORED HIGHER AL-HILAL OR AL-AHLI?"
        # question_entity_list = ["AL-HILAL", "AL-AHLI"]
        question_tokenized, attention_mask = self.question_processor.process_question(question, question_entity_list)
        question_embedding = self.question_processor.get_question_embedding(question_tokenized, attention_mask)

        kg_predictor = KGLinkPredictor(model, entity_to_id, relation_to_id)
        # result = self.process_retrieve_file(retrieve_file_path, question_id)#这个是根据fasttext获取的
        result_entities = []

        # for entity in retrieved_entity_data_list:
        #     candidate_entities = []
        #     related_entity = result[entity]

        #     for e in related_entity:

        # 并行处理每个实体的尾实体和头实体预测
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_entity = {}

            for entity_data in retrieved_entity_data_list:
                e = entity_data.entity_name
                candidate_entities = self.kg_handler.get_kg_entities()
                candidate_entities = list(set(candidate_entities))
                
                if not candidate_entities:
                    continue
                if e not in kg_predictor.entity_to_id.keys():
                    continue

                # 将尾实体预测任务添加到线程池
                future_to_entity[executor.submit(self.predict_tail_entities, e, question_embedding, candidate_entities, kg_predictor)] = ('tail', e)
                
                # 将头实体预测任务添加到线程池
                future_to_entity[executor.submit(self.predict_head_entities, e, question_embedding, candidate_entities, kg_predictor)] = ('head', e)

            # 处理线程池中完成的任务
            for future in concurrent.futures.as_completed(future_to_entity):
                task_type, entity = future_to_entity[future]
                try:
                    result = future.result()
                    if task_type == 'tail':
                        tail_entity_list = result
                        for tail_entity, score in tail_entity_list:
                            if score > 0:
                                result_entities.append(tail_entity)
                    elif task_type == 'head':
                        head_entity_list = result
                        for head_entity, score in head_entity_list:
                            if score > 0:
                                result_entities.append(head_entity)
                except Exception as e:
                    print(f"Error with entity {entity}: {e}")

        # for entity_data in retrieved_entity_data_list:
        #     e = entity_data.entity_name
        #     # candidate_entities = get_entities_with_hops(test_kg_path, e, hops=3)
        #     candidate_entities = self.kg_handler.get_kg_entities()
        #     candidate_entities = list(set(candidate_entities))
        #     # print(f'与{e}相联的实体：{candidate_entities}')
        #     if not candidate_entities:
        #         continue
        #     if e not in kg_predictor.entity_to_id.keys():
        #         continue
        #     tail_entity_list = kg_predictor.predict_tail_entity(e, question_embedding,candidate_entities,  model_name='ComplEx', top_m = 1)
        #     head_entity_list = kg_predictor.predict_head_entity(e, question_embedding, candidate_entities, model_name='ComplEx', top_m = 1)
            
        #     for tail_entity, score in tail_entity_list:
        #         if score > 0:
        #             result_entities.append(tail_entity)
        #         # print(f"Entity: {tail_entity}, Score: {score}")
        #     for head_entity, score in head_entity_list:
        #         if score > 0:
        #             result_entities.append(head_entity)
        #         # print(f"Entity: {head_entity}, Score: {score}")
    
        result_data_id_list = []
        result_entity_data_list = []
        for entity in result_entities:
            data_id = self.kg_handler.get_data_id_by_entity(entity)
            result_data_id_list += data_id
            
            entity_data_list = self.kg_config.KG.get_entity_by_entity(entity, data_id_list)
            result_entity_data_list += entity_data_list
        result_entity_data_str = ''
        for result_entity_data in result_entity_data_list:
            result_entity_data_str += f'{result_entity_data.entity_name}:{result_entity_data.data_id}\n'
        self.retrieve_config.retrieve_logger.log_info(f"2.基于kge检索到的实体:\n{result_entity_data_str}\n")
        result_entity_data_list.extend(retrieved_entity_data_list)
        # print(list(set(result_data_id_list)))
        return list(set(result_entity_data_list))

def parse_arguments():
    parser = argparse.ArgumentParser()

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
        "--KG_model", default='ComplEx', help="KGE model",
        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE']
    )
    parsed_args = parser.parse_args()
    return parsed_args

# if __name__ == "__main__":
    #下面这个是训练整个知识图谱
    # db_path = '/data/cmf/mmqa/kg/kg.db'
    # train_path = f'/data/cmf/mmqa/KGE/train.txt'
    # directory = os.path.dirname(train_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # model_save_path = f'/data/cmf/mmqa/KGE/trained_model.pkl'
    # triples_save_path = f'/data/cmf/mmqa/KGE/trained_triples.pkl'
    # trainer = KGTrainer(train_path, model_save_path, triples_save_path)
    # try:
    #     trainer.run(db_path)
    # except Exception as e:
    #     error_message = f"{str(e)}\n"
    #     print(f"An error occurred: {error_message}")  # 可选，打印错误信息

   
    
#     args = parse_arguments()
    
#     db_path = '/home/cmf/multiQA/KG_LLM/vector_store/test_data/kg/kg.db'
#     train_path = '/home/cmf/multiQA/KG_LLM/KGE/data/train.txt'
#     model_save_path = '/home/cmf/multiQA/KG_LLM/KGE/data/trained_model.pkl'
#     triples_save_path = '/home/cmf/multiQA/KG_LLM/KGE/data/trained_triples.pkl'

#     KG = KnowledgeGraph(db_path)
    
#     if args.dataset.lower == 'mmqa':
#         data_processor = MMQADataProcessor()

#     for idx in range(0, len(questions)):
#         questions,text_questions,table_questions,image_questions = data_processor.get_dev_information()
#         data_id_list = questions[idx]["image_doc_ids"] +questions[idx]["text_doc_ids"] + [questions[idx]["table_id"]]
#         kg_config = KgConfig(KG, KG.get_kg_entity_names(data_id_list), KG.get_entity_link(idx), data_id_list)

#         main_program = KGERetrival(args, db_path, kg_config, train_path, model_save_path, triples_save_path)
#         main_program.run()
