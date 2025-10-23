from prompt.prompt_generator import TextEntitiesPrompt_New, TextRelationshipPrompt_New, TextEntitiesPrompt, TextRelationshipPrompt
from utils.proprecss_llm_answer import extract_llm_text_entities_new, extract_llm_text_relationships
from utils.utils import combine_dictionaries, write_json, simply_combine_dict, combine_relations_dict
import concurrent.futures
from typing import List, Dict, Tuple
import time
import re

class TextEntityExtractor:
    def __init__(self, args, LLM, max_num_tries=3):
        self.args = args
        self.LLM = LLM
        self.max_num_tries = max_num_tries

    def extract_entities(self, text, text_id, is_title=False):
        entities = {}
        try_count = 0
        max_num_tries = self.max_num_tries
        if is_title:
            max_num_tries = 2
        # print(is_title, ':',text)
        while max_num_tries:
            try:
                max_num_tries -= 1
                print(max_num_tries)
                prompt = self._generate_prompt(text)
                self.LLM.init_message()
                result, flag = self.LLM.predict(self.args, prompt)
                # print(f'llm:{result}')
                # text_entities = self._extract_text_entities(result)
                text_entities = self.extract_text_entities(result)
                # print(f'text entities:{text_entities}')
                entities = self._combine_entities(entities, text_entities)
            
                
            except Exception as e:
                try_count += 1
                
                if try_count>3 and max_num_tries>0 and not entities:
                    max_num_tries = 0
                    
                    return entities, False

                elif try_count<=3:
                    max_num_tries += 1
                
                else:
                    if not entities:
                        flag = False
                    return entities, flag
        
        
        if not entities:
            flag = False
        else:
            flag = True
        return entities, flag

    
    def extract_entities_batch(self, text_id_mapping, is_title=False):
        """
        批量处理多个文本，利用 LLaMA 模型的批量预测能力。
        :param texts: 文本列表
        :param text_ids: 对应文本的ID列表
        :param is_title: 是否为标题的标志
        :return: 字典, key为文本ID, value为该文本的实体
        """
        all_entities = {}
        text_datas = {}
        try_count = 0
        max_num_tries = self.max_num_tries

        if is_title:
            max_num_tries = 2

        prompts = [self._generate_prompt(text_id_data[0]) for text_id_data in text_id_mapping]

        start_time = time.time()
        while max_num_tries:
            try:
                max_num_tries -= 1

                outputs = self.LLM.predict_batch(self.args, prompts)  # 批量处理多个文本

                for i, (text ,text_id, text_data) in enumerate(text_id_mapping):
                    text_result = outputs[i].strip()  # 清除多余的空白字符
                    # print('llm entities:\n', text_result)
                    # text_entities = self._extract_text_entities(text_result)
                    text_entities = self.extract_text_entities(text_result)
                    # print('extract entities:\n', text_entities)
                    existing_entities = all_entities.get(text_id, {})
                    all_entities[text_id] = self._combine_entities(existing_entities, text_entities)
                    # print(f'all_entities[{text_id}]:\n{all_entities[text_id]}')

            except Exception as e:
                try_count += 1
                if try_count > 3 and max_num_tries > 0 and not all_entities:
                    print(f"Error during extraction: {text_result} {e}")
                    max_num_tries = 0
                    break
                elif try_count <= 3 :
                    max_num_tries += 1
                else:
                    break
        
        # print('batch extract entities result:')
        # print(time.time()- start_time, all_entities)
        for (text, text_id, text_data) in text_id_mapping:
            text_datas[text_id] = text_data
            if text_id not in all_entities or not all_entities[text_id]:
                all_entities[text_id] = {"status": False}
                

        return all_entities, text_datas

    def _generate_prompt(self, text):
        prompt_generator = TextEntitiesPrompt_New() 
        return prompt_generator.create_prompt(text)

    def _extract_text_entities(self, result):
        return extract_llm_text_entities_new(result)
    
    def extract_text_entities(self, result):
        pattern = re.compile(
            r'entity name:\s*(?P<name>.*?)\s*'
            r'entity type:\s*(?P<type>.*?)\s*'
            r'description:\s*(?P<description>[^\n]+)',
            re.IGNORECASE | re.DOTALL
        )

        entities = {}
        # print(pattern.finditer(result))
        for match in pattern.finditer(result):
            entity_name = match.group("name").strip()
            entity_type = match.group("type").strip()
            description = match.group("description").strip()
            # print(entity_name, entity_type, description)

            if entity_name not in entities:
                entities[entity_name] = {}

            entities[entity_name] = {
                "Type": entity_type,
                "Description": description
            }
        return entities


    def _combine_entities(self, existing_entities, new_entities):
        # print('进入合并')
        # print(f'合并结果:{simply_combine_dict(new_entities, existing_entities)}')
        return simply_combine_dict(new_entities, existing_entities)



class TextRelationExtractor:
    def __init__(self, args, LLM, max_num_tries=3):
        self.args = args
        self.LLM = LLM
        self.max_num_tries = max_num_tries

    def extract_relations(self, text, text_id, entities):
        # print('text:', text)
        relations = {}
        try_count = 0
        max_num_tries = self.max_num_tries
        print(f'进入循环前max_num:{max_num_tries}')

        while max_num_tries > 0:
            # try:
            max_num_tries -= 1
            
            prompt = self._generate_prompt(text, entities)
            self.LLM.init_message()
            # print('llm预测开始')
            result, flag = self.LLM.predict(self.args, prompt)
            # print(f'llm result:{result}')
            # text_relations = self._extract_text_relations(result)
            text_relations = self.extract_text_relations(result)
            # print(f'relation result:{text_relations}')
            relations = self._combine_relations(relations, text_relations)
            # print(f'此时max_num:{max_num_tries}')
                
            # except Exception as e:
            #     try_count += 1
                
            #     if try_count>3 and max_num_tries>0 and not relations:
            #         max_num_tries = 0
                    
            #         return relations, False

            #     elif try_count<=3:
            #         max_num_tries += 1
                
            #     else:
            #         if not relations:
            #             flag = False
            #         return relations, flag
        
        # print(f'final relations:{relations}')
        if not relations:
            flag = False
        else:
            flag = True
        return relations, flag

    
    def extract_relations_batch(self, text_id_mapping, entities_batch_list):
        """
        批量处理多个文本，利用 LLaMA 模型的批量预测能力。
        :param texts: 文本列表
        :param text_ids: 对应文本的ID列表
        :param is_title: 是否为标题的标志
        :return: 字典, key为文本ID, value为该文本的实体
        """

        print('开始预测')
        all_relations = {}
        text_datas = {}
        try_count = 0
        max_num_tries = self.max_num_tries

        prompts = [
            self._generate_prompt(text_id_data[0], entities)
            for text_id_data, entities in zip(text_id_mapping, entities_batch_list)
        ]

        start_time = time.time()
        while max_num_tries:
            try:
                max_num_tries -= 1
                print(max_num_tries)

                outputs = self.LLM.predict_batch(self.args, prompts)  # 批量处理多个文本

                for i, (text ,text_id, text_data) in enumerate(text_id_mapping):
                    text_result = outputs[i].strip()  # 清除多余的空白字符
                    print('llm relations:\n', text_result)
                    text_relations = self.extract_text_relations(text_result)
                    print('extract relations:\n', text_relations)
                    existing_relations = all_relations.get(text_id, {})
                    all_relations[text_id] = self._combine_relations(existing_relations, text_relations)
                    print(f'all_relations[{text_id}]:\n{all_relations[text_id]}')

            except Exception as e:
                try_count += 1
                if try_count > 3 and max_num_tries > 0 and not all_relations:
                    print(f"Error during extraction:  {e}")
                    max_num_tries = 0
                    break
                elif try_count <= 3 :
                    max_num_tries += 1
                else:
                    break
        for (text, text_id, text_data) in text_id_mapping:
            print(f'{text_id}结果：{all_relations[text_id]}')
            text_datas[text_id] = text_data
            if text_id not in all_relations or not all_relations[text_id]:
                all_relations[text_id] = {"status": False}
                

        return all_relations, text_datas

    def _generate_prompt(self, text, entities):
        prompt_generator = TextRelationshipPrompt_New()
        return prompt_generator.create_prompt(text, entities)

    def extract_text_relations(self, result):
        # pattern = re.compile(
        #     r'head entity:(?P<head>.*?)\s+'
        #     r'tail entity:(?P<tail>.*?)\s+'
        #     r'relation:(?P<relation>.*?)\s+'
        #     r'description:(?P<description>.*?)\s+'
        #     r'strength:(?P<strength>.*?)$',
        #     re.MULTILINE | re.IGNORECASE
        # )

        pattern = re.compile(
            r'head entity:\s*(?P<head>.*?)\s+'
            r'tail entity:\s*(?P<tail>.*?)\s+'
            r'relation:\s*(?P<relation>.*?)\s+'
            r'description:\s*(?P<description>.*?)\s+'
            r'strength:\s*(?P<strength>\d+)',
            re.IGNORECASE | re.DOTALL
        )

        relationships = {}
        # print(pattern.finditer(result))
        for match in pattern.finditer(result):
            head = match.group("head").strip()
            tail = match.group("tail").strip()
            relation = match.group("relation").strip()
            description = match.group("description").strip()
            strength = match.group("strength").strip()
            # print(head, tail, relation, description, strength)

            if head not in relationships:
                relationships[head] = {}

            relationships[head][tail] = {
                "Relation": relation,
                "Description": description,
                "Strength": strength
            }
        return relationships

    

    def _extract_text_relations(self, result):
        return extract_llm_text_relationships(result)

    def _combine_relations(self, existing_relations, new_relations):
        # print('进入relation合并')
        # print(f'existing_relations:\n{existing_relations}')
        # print(f'new_relations:\n{new_relations}')
        # print(f'合并结果：{combine_relations_dict(existing_relations, new_relations)}')
        return combine_relations_dict(existing_relations, new_relations)


    
def extract_text_entities(result):
    pattern = re.compile(
        r'entity name:\s*(?P<name>.*?)\s*'
        r'entity type:\s*(?P<type>.*?)\s*'
        r'description:\s*(?P<description>[^\n]+)',
        re.IGNORECASE | re.DOTALL
    )

    entities = {}
    # print(pattern.finditer(result))
    for match in pattern.finditer(result):
        entity_name = match.group("name").strip()
        entity_type = match.group("type").strip()
        description = match.group("description").strip()
        print(entity_name, entity_type, description)

        if entity_name not in entities:
            entities[entity_name] = {}

        entities[entity_name] = {
            "Type": entity_type,
            "Description": description
        }
    return entities

def extract_text_relations(result):

    pattern = re.compile(
        r'head entity:\s*(?P<head>.*?)\s+'
        r'tail entity:\s*(?P<tail>.*?)\s+'
        r'relation:\s*(?P<relation>.*?)\s+'
        r'description:\s*(?P<description>.*?)\s+'
        r'strength:\s*(?P<strength>\d+)',
        re.IGNORECASE | re.DOTALL
    )

    relationships = {}
    print(pattern.finditer(result))
    for match in pattern.finditer(result):
        head = match.group("head").strip()
        tail = match.group("tail").strip()
        relation = match.group("relation").strip()
        description = match.group("description").strip()
        strength = match.group("strength").strip()
        print(head, tail, relation, description, strength)

        if head not in relationships:
            relationships[head] = {}

        relationships[head][tail] = {
            "Relation": relation,
            "Description": description,
            "Strength": strength
        }
    return relationships
result ='''
llm:Here are the extracted entities:

entity name:Auguste Rodin
entity type:Person
description:French sculptor and artist, known for his works such as "The Thinker".

entity name:Baudelaire
entity type:Person
description:French poet, best known for his collection of poems "Les Fleurs du Mal".

entity name:Les Fleurs du Mal
entity type:Book
description:A collection of poems by Charles Baudelaire.

entity name:Gustave Courbet
entity type:Person
description:French painter and sculptor, known for his works such as "Le Sommeil".

entity name:Le Sommeil
entity type:Artwork
description:A painting by Gustave Courbet depicting two women asleep after love-making.

entity name:Mademoiselle de Maupin
entity type:Book
description:A novel by Théophile Gautier, published in 1835.

entity name:Théophile Gautier
entity type:Person
description:French writer and poet, known for his novels such as "Mademoiselle de Maupin".

entity name:Delphine et Hippolyte
entity type:Book
description:A short story by Charles Baudelaire, part of the collection "Les Fleurs du Mal".
'''
# print(extract_text_entities(result))

result = '''
Here are the possible relationships between the entities:


head entity: Chip E.
tail entity: He
relation: Same Person
description: Both names refer to the same individual
strength: 10
head entity: Chip E's 1985 recording It's House
tail entity: Work
relation: Is A
description: The recording is a work of music
strength: 9

**Relationship 3**
head entity: Chip E's 1985 recording It's House
tail entity: Recording
relation: Is A
description: The recording is a specific instance of recorded music
strength: 8

**Relationship 4**
head entity: Chip E.
tail entity: Person
relation: Is A
description: Chip E. is a person
strength: 7

**Relationship 5**
head entity: House music
tail entity: Concept
relation: Is A
description: House music is a concept or genre of electronic music
strength: 6

Let me know if you'd like me to generate more relationships!

'''


# print(extract_text_relations(result))