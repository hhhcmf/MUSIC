import sqlite3
from KG_LLM.retrieve import RetrieveRelationsPhase, RetrieveEntitiesPhase
from utils.utils import get_image_path, extract_table_names
from models.qa.imageqa import Image_qa
from config.config import IMAGE_PATH, MMQA_DB
from data.MMQA.preprocess import get_tableId_by_name, get_tableCol_by_col, get_cols_by_table_id
def get_entity_information(entity_data, question, question_embedding, LLM, KG, kg_entities_name, args, logger, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_triple_k):
    retrieve_relations_phase = RetrieveRelationsPhase(LLM, args, KG, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, top_triple_k, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key)
    entity_name = entity_data.entity_name
    data_id = entity_data.data_id
    data_type = entity_data.type
    title = entity_data.title
    is_column = entity_data.is_column
    is_column_value = entity_data.is_column_value
    isImageTitle = entity_data.isImageTitle
    table_name = entity_data.table_name
    columns = entity_data.columns
    column = entity_data.column
    description = entity_data.description
    text_data = entity_data.text
    result = ''
    if data_type == 'table':
        name = table_name[len(title)+1:]
        result += f'Data Id: {data_id}\n'
        result += 'Modality Type: table\n'

        table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
        
        if is_column_value:  
            result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"
        elif is_column:  # If it's a column name
            result += f'Data Type: Column name\n'           
        elif entity_name == title:  # General table information
            result += f'Data Type: Table Title\n'
        else:
            result += f'Data Type: Table Name\n'
        result += table_desp
        
    elif data_type == 'image':
        result += f'Data Id: {data_id}\n'
        # result += f'Entity Name: {entity_name}\n'
        result += 'Modality Type: image\n'
        if isImageTitle:
            result += f'Data Type: Image Title\n'
            # result += f'Image Description:{text_data}\n'
        else: 
            result += f'Date Type: Object\n'
        result += f'Image Title:{title}\n'
        result += f"Triples: ({title}, HAS, {entity_name})\n"
        
    else:
        # pos += 1
        # result += f'{pos})'
        result += f'Data Id: {data_id}\n'
        result += 'Modality Type: text\n'
        result += f'Entity Description: {description}\n'
        result += f'Text data:{text_data}\n'
        result += f'Triples:\n'
        triples = KG.get_triples_by_entity(entity_name)
        if len(triples)>10:
            triples = retrieve_relations_phase.retrieve(question, question_embedding, triples)
        
        for triple in triples:
            head_entity = triple.head_entity
            tail_entity = triple.tail_entity
            relation = triple.relation
            description = triple.description
            result += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
    return result

def QueryEntity(entity_name, question, question_embedding, entity_link_hash, LLM, KG, kg_entities_name, args, logger, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, entity_index, entity_id_to_key, entity_desp_index, entity_desp_id_to_key, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_entity_k, top_triple_k):
    '''
    Find entity and their data in kg by entity_name
    '''
    # result = f'API:QueryEntity Input:{entity_name} Output:\n'
    retrieve_relations_phase = RetrieveRelationsPhase(LLM, args, KG, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, top_triple_k, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key)
    entity_link_keys = entity_link_hash.keys()
    entities_dict = {} 
    table_entity_data = []
    image_entity_data = []
    text_entity_data = []
    entity_name_S = entity_name
    entity_name_THE = entity_name
    entity_name_start_space = entity_name
    entity_name_end_space = entity_name
    if entity_name_start_space.startswith(' '):
        entity_name_start_space = entity_name_start_space[1:]
    if entity_name_end_space.startswith(' '):
        entity_name_end_space = entity_name_end_space[1:]

    if entity_name.endswith('S'):
        entity_name_S = entity_name[:-1]
    if entity_name.startswith('THE'):
        entity_name_THE = entity_name[3:]
        while entity_name_THE.startswith(' '):
            entity_name_THE = entity_name_THE[1:]
    entity_name_alias = [entity_name, f'THE {entity_name}', entity_name_end_space, entity_name_start_space, f'{entity_name}S', entity_name_S, entity_name_THE]
    find_flag = False
    for entity_name in entity_name_alias:
        if entity_name in kg_entities_name:
            find_flag = True
            entities_dict = {}
            entities_dict[entity_name] = []
            entity_name_list = [entity_name]
            entity_data_list = KG.get_entity_by_entity(entity_name)
            entities_dict[entity_name] = entity_data_list
            break
    if not find_flag:
        retrieve_entities_phase = RetrieveEntitiesPhase(LLM, args, KG, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, top_entity_k, entity_index, entity_id_to_key, entity_desp_index, entity_desp_id_to_key, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_triple_k)
        entities_dict = retrieve_entities_phase.retrieve(question, question_embedding, entity_name, entity_link_hash)
    result = f'Entities related to {entity_name}:\n'
    idx = 0
    if not entities_dict:
        result += f'There are no entities related to {entity_name} in the knowledge graph and need to obtain information related to {entity_name} by searching for information on other entities.'
        return table_entity_data, image_entity_data, text_entity_data, result, False
    for entity_name, entity_data_list in entities_dict.items():
        idx += 1
        result += f'({idx})'
        result += f'Entity Name: {entity_name}\n'
       
        pos = 0 
        for entity_data in entity_data_list:
            data_id = entity_data.data_id
            data_type = entity_data.type
            title = entity_data.title
            is_column = entity_data.is_column
            is_column_value = entity_data.is_column_value
            isImageTitle = entity_data.isImageTitle
            table_name = entity_data.table_name
            columns = entity_data.columns
            column = entity_data.column
            description = entity_data.description
            text_data = entity_data.text
            print(data_type)
            print(data_id)
            if data_type == 'table':
                if entity_data not in table_entity_data:
                    table_entity_data.append(entity_data)
                pos += 1
                result += f' {pos})'

                name = table_name[len(title)+1:]
                result += f'Data Id: {data_id}\n'
                result += 'Modality Type: table\n'

                table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
                
                if is_column_value:  
                    result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"
                elif is_column:  # If it's a column name
                    result += f'Data Type: Column name\n'
                
                elif entity_name == title:  # General table information
                    result += f'Data Type: Table Title\n'
                else:
                    result += f'Data Type: Table Name\n'
                result += table_desp
                if (entity_name, data_id) in entity_link_keys:
                    count = 0
                    for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                        if entity_link_data.type == 'image' and entity_link_data not in image_entity_data:
                            image_entity_data.append(entity_link_data)
                        if entity_link_data.type == 'text' and entity_link_data not in text_entity_data:
                            text_entity_data.append(entity_link_data)
                        result += 'Alias Entities:\n'
                        count += 1
                        result += f'  {count})'
                        result += f'Entity Name: {entity_link_data.entity_name}\n'
                        result += get_entity_information(entity_link_data, question, question_embedding, LLM, KG, kg_entities_name, args, logger, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_triple_k)
                        
                
            elif data_type == 'image':
                if entity_data not in image_entity_data:
                    image_entity_data.append(entity_data)
                pos += 1
                result += f'{pos})'
                result += f'Data Id: {data_id}\n'
                # result += f'Entity Name: {entity_name}\n'
                result += 'Modality Type: image\n'
                if isImageTitle:
                    result += f'Data Type: Image Title\n'
                    # result += f'Image Description:{text_data}\n'
                else: 
                    result += f'Date Type: Object\n'
                result += f'Image Title:{title}\n'
                result += f"Triples: ({title}, HAS, {entity_name})\n"
                
                if (entity_name, data_id) in entity_link_keys:
                    count = 0
                    for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                        if entity_link_data.type == 'table' and entity_link_data not in table_entity_data:
                            table_entity_data.append(entity_link_data)
                        if entity_link_data.type == 'text' and entity_link_data not in text_entity_data:
                            text_entity_data.append(entity_link_data)
                        result += 'Alias Entities:\n'
                        count += 1
                        result += f'  {count})'
                        result += f'Entity Name: {entity_link_data.entity_name}\n'
                        result += get_entity_information(entity_link_data, question, question_embedding, LLM, KG, kg_entities_name, args, logger, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_triple_k)
                        
            else:
                if entity_data not in text_entity_data:
                    text_entity_data.append(entity_data)
                pos += 1
                result += f'{pos})'
                result += f'Data Id: {data_id}\n'
                # result += f'Entity Name: {entity_name}\n'
                result += 'Modality Type: text\n'
                result += f'Entity Description: {description}\n'
                result += f'Text data:{text_data}\n'
                result += f'Triples:\n'
                # result += f'{entity_name} belongs to the text modality data; the triples format is (entity, relation, entity):<description of triples>. All related triples are:'
                triples = KG.get_triples_by_entity(entity_name)
                if len(triples)>10:
                    triples = retrieve_relations_phase.retrieve(question, question_embedding, triples)
                
                for triple in triples:
                    head_entity = triple.head_entity
                    tail_entity = triple.tail_entity
                    relation = triple.relation
                    description = triple.description
                    result += f'({head_entity}, {relation}, {tail_entity}):{description}\n'
                if (entity_name, data_id) in entity_link_keys:
                    count = 0
                    for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                        if entity_link_data.type == 'image' and entity_link_data not in image_entity_data:
                            image_entity_data.append(entity_link_data)
                        if entity_link_data.type == 'table'and entity_link_data not in table_entity_data:
                            table_entity_data.append(entity_link_data)
                        result += 'Alias Entities:\n'
                        count += 1
                        result += f'  {count})'
                        result += f'Entity Name: {entity_link_data.entity_name}\n'
                        
                        result += get_entity_information(entity_link_data, question, question_embedding, LLM, KG, kg_entities_name, args, logger, retrieve_logger, embedding_model_name, embed_model, embed_tokenizer, triple_index, triple_id_to_key, triple_desp_index, triple_desp_id_to_key, top_triple_k)

    return table_entity_data, image_entity_data, text_entity_data, result, True



def get_answer_alias_entity_information(entity_data, text_flag, text_answer_flag, image_title_flag):
    
    entity_name = entity_data.entity_name
    data_id = entity_data.data_id
    data_type = entity_data.type
    title = entity_data.title
    is_column = entity_data.is_column
    is_column_value = entity_data.is_column_value
    isImageTitle = entity_data.isImageTitle
    table_name = entity_data.table_name
    columns = entity_data.columns
    column = entity_data.column
    description = entity_data.description
    text_data = entity_data.text
    
    result = ''
    
    if data_type == 'table':
        name = table_name[len(title)+1:]
        result += 'Modality Type: table\n'

        table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
        sql_answer = ''
        if is_column_value:  
            result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"  
            select_columns = columns.replace('@#', ',')
            sql_query = f'SELECT {select_columns} FROM {table_name}'
            conditions = {}
            conditions[entity_name] = column
            conn = sqlite3.connect(MMQA_DB)
            cursor = conn.cursor()
            sql_answer, _ = QueryTable(table_name, select_columns, conditions, cursor)
            cursor.close()
            conn.close()
        elif is_column:  # If it's a column name
            result += f'Data Type: Column name\n'           
        elif entity_name == title:  # General table information
            result += f'Data Type: Table Title\n'
        else:
            result += f'Data Type: Table Name\n'
        result += table_desp
        result += sql_answer
        
    elif data_type == 'image':
        if isImageTitle:
            image_title_flag = True
            result += 'Modality Type: image\n'
            result += f'Image Title:{title}\n'
        
    else:
        # pos += 1
        # result += f'{pos})'
        text_flag = True
        result += 'Modality Type: text\n'
        result += f'Entity Description: {description}\n'
        result += f'Text data:{text_data}\n'
       
    return result, text_flag, text_answer_flag, image_title_flag

def get_answer_entity(entity_name, entity_link_hash, KG, kg_entities_name):
    entity_link_keys = entity_link_hash.keys()
    entities_dict = {} 
    table_entity_data = []
    image_entity_data = []
    text_entity_data = []
    if entity_name in kg_entities_name:
        entities_dict = {}
        entities_dict[entity_name] = []
        entity_name_list = [entity_name]
        entity_data_list = KG.get_entity_by_entity(entity_name)
        entities_dict[entity_name] = entity_data_list
    result = ''
    image_title_flag = False
    text_flag = False
    text_answer_flag = True
    for entity_name, entity_data_list in entities_dict.items():
        
        result += f'Entity Name: {entity_name}\n'
       
        pos = 0 
        for entity_data in entity_data_list:
            data_id = entity_data.data_id
            data_type = entity_data.type
            title = entity_data.title
            is_column = entity_data.is_column
            is_column_value = entity_data.is_column_value
            isImageTitle = entity_data.isImageTitle
            table_name = entity_data.table_name
            columns = entity_data.columns
            column = entity_data.column
            description = entity_data.description
            text_data = entity_data.text

            if data_type == 'table':
                
                pos += 1
                result += f' ({pos})'

                name = table_name[len(title)+1:]
                result += 'Modality Type: table\n'

                table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
                sql_answer = ''
                if is_column_value:  
                    result += f"Data Type: Column Value\nBelonging Column Name:{column}\n"  
                    select_columns = columns.replace('@#', ',')
                    sql_query = f'SELECT {select_columns} FROM {table_name}'
                    conditions = {}
                    conditions[entity_name] = column
                    conn = sqlite3.connect(MMQA_DB)
                    cursor = conn.cursor()
                    sql_answer, _ = QueryTable(table_name, select_columns, conditions, cursor)
                    cursor.close()
                    conn.close()
                    

                elif is_column:  # If it's a column name
                    result += f'Data Type: Column name\n'
                
                elif entity_name == title:  # General table information
                    result += f'Data Type: Table Title\n'
                else:
                    result += f'Data Type: Table Name\n'
                result += table_desp
                result += sql_answer
                if (entity_name, data_id) in entity_link_keys:
                    count = 0
                    for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                        if entity_link_data.entity_name != entity_name:
                            result += 'Alias Entities:\n'
                            count += 1
                            result += f'  {count})'
                            result += f'Entity Name: {entity_link_data.entity_name}\n'
                            output, text_flag, text_answer_flag, image_title_flag = get_answer_alias_entity_information(entity_link_data, text_flag, text_answer_flag, image_title_flag)
                            result += output
                
            elif data_type == 'image':
                if isImageTitle:
                    image_title_flag = True
                    pos += 1
                    result += f'({pos})'
                    
                    result += 'Modality Type: image\n'
                    result += f'Image Title:{title}\n'
                    
                    if (entity_name, data_id) in entity_link_keys:
                        count = 0
                        for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                            if entity_link_data.entity_name != entity_name:
                                result += 'Alias Entities:\n'
                                count += 1
                                result += f'  {count})'
                                result += f'Entity Name: {entity_link_data.entity_name}\n'
                                output, text_flag, text_answer_flag, image_title_flag = get_answer_alias_entity_information(entity_link_data, text_flag, text_answer_flag, image_title_flag)
                                result += output
                        
            else:
                text_flag = True
                pos += 1
                result += f'({pos})'
                result += 'Modality Type: text\n'
                result += f'Entity Description: {description}\n'
                result += f'Text data:{text_data}\n'
                
                if (entity_name, data_id) in entity_link_keys:
                    count = 0
                    for entity_link_data in entity_link_hash[(entity_name, data_id)]:
                        if entity_link_data.entity_name != entity_name:
                            result += 'Alias Entities:\n'
                            count += 1
                            result += f'  {count})'
                            result += f'Entity Name: {entity_link_data.entity_name}\n'
                            output, text_flag, text_answer_flag, image_title_flag = get_answer_alias_entity_information(entity_link_data, text_flag, text_answer_flag, image_title_flag)
                            result += output

    return result, text_flag, text_answer_flag, image_title_flag



    


