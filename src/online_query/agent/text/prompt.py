from KG_LLM.online_query.agent.base import BasePromptGenerator
class TextPrompt(BasePromptGenerator):
    def __init__(self, kg_config, example=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example = example
        self.kg_config = kg_config

    def _get_information(self, text_entity_data_list):
        entity_information, text_information = '', ''
        for i, entity_data in enumerate(text_entity_data_list):
            title = entity_data.title
            text = entity_data.text
            triples = self.kg_config.KG.get_triples_by_entity(entity_data.entity_name, entity_data.data_id)
            entity_information += f'({i+1})Entity Name:{entity_data.entity_name}\nData Id:{entity_data.data_id}\nTitle:{entity_data.title}\nEntity Description:{entity_data.description}\nTriples:\n'
            if len(text)<200:
                entity_information += f'Text:{text}\n'
            strong_triple_list = []
            for triple in triples:
                # print('text prompt:', triple.strength)
            #     if int(triple.strength) > 5:
            #         strong_triple_list.append(triple)
            
            # if len(strong_triple_list) < 8:
                entity_information += f'({triple.head_entity}, {triple.relation}, {triple.tail_entity}):{triple.description}\n'
            # else:
                
            
            entity_information += '\n'
        return entity_information, text_information

    

    def _create_judge_prompt(self, question, userful_information, text_entity_data_list, *args, **kwargs):
        entity_information, _ = self._get_information(text_entity_data_list)
        prompt = f'''
You are a data analysis expert. Given a natural language question, known knowledge, and some text modality entity information, please identify the text modality entities related to the question based on the provided information. Output the entity names and their corresponding data IDs for the text modality entities relevant to the question.If there are no relevant text modality entities, please output "no".
For text-modality entities, each entity's information contains the source data id, a description of the entity, related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
Every entity may have alias entities,alias entities refer to different representations or names of the same underlying entity. These aliases represent the same entity but are stored with different entity names, modalities (such as image, text, or table), or related information. The purpose of alias entities is to acknowledge that multiple representations, across various forms or names, still refer to the same core entity.

Question: {question}
Known knowledge:{userful_information}
Entity information:
{entity_information}

Please find the text-modality entities related to the question and output the entity name and data ID.If there are no relevant text modality entities, please output "no".
Please output in the following format.
'no' or
entity name:<entity name> data id:<data id>
entity name:<entity name> data id:<data id>
...
entity name:<entity name> data id:<data id
'''
        return prompt

    def _create_reasoning_prompt(self, userful_information, question, title, text, *args, **kwargs):
        prompt = f'''
Obtained data:
{userful_information}

Question:{question}

Please analyze the obtained data. Your task is to extract sentences from the passage given in the following passage data that are relevant to the question and helpful in answering the question. Note that it is sentences, not words.
passage data:
title(Indicate that the passage is themed around the title, and the content in the passage is related to the title):{title}
passage:{text}

Please output in the following format.
output:<related text snippets> or 'no related text snippet'
        '''
        return prompt
    
    def create_vote_prompt(self, userful_information, question, texts):
        passages = ''
        idx = 1
        for text in texts:
            passages += f'{idx}. {text}\n'
            idx += 1


        prompt += f'''
Question:{question}
Obtained data:
{userful_information}
Please analyze the data already obtained.Your task is to find a passage from the given passages that is relevant to the question and provides the most comprehensive background.
passages:
{passages}
Please output in the following format.
output:<related text snippets>    
        '''
        return prompt