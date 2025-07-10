from online_query.agent.base import BasePromptGenerator
class TablePrompt(BasePromptGenerator):
    def __init__(self, example=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example = example
    
    def _get_information(self, table_entity_data_list):
        entity_information, table_information = '', ''
        conditions = {}
        for i, entity_data in enumerate(table_entity_data_list):
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
            table_columns = columns.replace('@#', ',')

            if is_column_value:
                belong_column = entity_data.column
                column_value = entity_data.entity_name
                conditions[column_value] = belong_column
            
            entity_information += f'({i+1})'
            entity_information += f'Entity Name:{entity_name}\n'
            if title:
                name = table_name[len(title)+1:]
                table_desp = f'Table Title:{title}\nTable Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\nTable Description: This table records {name} of {title}.\n'
            else:
                table_desp = f'Table Name:{table_name}\nColumn Name:{columns} (columns are separated by "@#")\n'
            
            entity_information += f'Data Id: {data_id}\n'
            entity_information += 'Modality Type: table\n'        
            if is_column_value:  
                entity_information += f"Data Type: Column Value\nBelonging Column Name:{column}\n"
            elif is_column:  # If it's a column name
                entity_information += f'Data Type: Column name\n'           
            elif entity_name == title:  # General table information
                entity_information += f'Data Type: Table Title\n'
            else:
                entity_information += f'Data Type: Table Name\n'
            entity_information += table_desp
        
        conditions_str = ''
        for key, value in conditions.items():
            conditions_str += f'"{key}:{value},\n"'
        if conditions_str.endswith(',\n'):
            conditions_str = conditions_str[:-2]
        table_information += f'Table Name:{table_name}\nTable Column:{columns}(columns are separated by "@#")\nConditions:\n{{{conditions_str}}}\n'

        return entity_information, table_information
    
    def _get_table_information(self, table_entity_data_list):
        idx = 0
        table_data_information = ''
        table_name = table_entity_data_list[0].table_name
        columns = table_entity_data_list[0].columns
        title = table_entity_data_list[0].title

        for i, entity_data in enumerate(table_entity_data_list):
            entity_name = entity_data.entity_name
            data_id = entity_data.data_id
            data_type = entity_data.type
            title = entity_data.title
            is_column = entity_data.is_column
            is_column_value = entity_data.is_column_value
            isImageTitle = entity_data.isImageTitle
            
            column = entity_data.column
            description = entity_data.description
            text_data = entity_data.text
            table_columns = columns.replace('@#', ',')
            
            if is_column_value:
                belong_column = entity_data.column
                cell_value = entity_data.entity_name
                idx+=1 
                table_data_information += f'{idx}.cell value:{cell_value} corresponding column:{belong_column}\n'
        

        name = table_name[len(title)+1:]
        table_information = f'Table Information:\nTable Title(the title is a descriptive expression of the table, not the table name or specific values within the table):{title}\n'
        table_information += f'Table Name:{table_name}\nColumn Name:{columns}\nTable Description: This table records {name} of {title}.\n'
        if table_data_information:
            table_information += f'\nPartial Data of the Table (only some data is presented):\n{table_data_information}\n'
        return table_information

    def _create_judge_prompt(self, question, userful_information, table_entity_data_list, *args, **kwargs):
        table_information = self._get_table_information(table_entity_data_list)
        prompt = f'''
You are a data analysis expert, and I will provide you with a natural language question, known knowledge, table information, and some partial table data (the table data may not be fully provided, and only part of the data will be shown).
Your task is to determine whether this table is related to the question based on the given information. If it is relevant to the question, output "yes," otherwise output "no" and provide the reason.

The table information contains the table's title (which is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, and a description of the table.
The internal table data format is as follows:
cell value: <cell value> 
corresponding column: <the name of the corresponding column>
(This means the cell value and the column name to which this value belongs.)

Known knowledge:{userful_information}

Now I will provide you with a natural language question, table information, and some partial table data (the table data may not be fully provided, and only part of the data will be shown).
Question:{question}
{table_information}

Output:Yes or No
'''
        return prompt

    def _create_reasoning_prompt(self, question, userful_information, table_entity_data_list, *args, **kwargs):
        #table_information
        # table name:132412
        # column names: State, Times hosted, First, Last
        # filtering conditions:State=South Carolina

        table_information = self._get_table_information(table_entity_data_list)
        prompt = f'''
You are a data analysis expert. I will provide a natural language question, known knowledge, table information, and some partial table data (the table data is not fully provided).
Our task is to identify the relevant table records that can help answer the question. Your job is to analyze the given information, identify all the column names related to the question (as there may be more than one), and the filtering conditions, so as to find the relevant table records.

Now I will provide the known information, which includes three types of data:
1. **Knowledge Graph Information:**
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.

2. **Image Data**: Presented in a Q&A format, including image data id, image title, image-based questions, and their answers.Each image title is an entity.

3. **Passages**: Includes text data id ,title and the text content.

{userful_information}

Next, I will provide the natural language question, table information, and some partial table data (the table data may not be fully provided, and only part of the data will be shown).
The table information contains the table's title (which is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, and a description of the table.
The internal table data format is as follows:
cell value: <cell value> 
corresponding column: <the name of the corresponding column>
(This means the cell value and the column name to which this value belongs.)

{table_information}
Question:{question}


        '''

        prompt += '''
Our task is to identify the relevant table records that can help answer the question. Your job is to analyze the given information, identify all the column names related to the question (as there may be more than one), and the filtering conditions, so as to find the relevant table records.
Please output the result in the following JSON format(no any explanation, no ```json```):
{
    "QueryTable-1":{   
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or {
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    },
    "QueryTable-2":
    {   
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or { 
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    },
    ....
    'QueryTable-n':{
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or { 
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    }

}
        '''
        return prompt
    def create_pipeline_reason_prompt(self, question, userful_information, table_entity_data_list, *args, **kwargs):
        #table_information
        # table name:132412
        # column names: State, Times hosted, First, Last
        # filtering conditions:State=South Carolina

        table_information = self._get_table_information(table_entity_data_list)
        prompt = f'''
You are a data analysis expert. I will provide a natural language question, known knowledge, table information, and some partial table data (the table data is not fully provided).
Our task is to identify the relevant table records that can help answer the question. Your job is to analyze the given information, identify all the column names related to the question (as there may be more than one), and the filtering conditions, so as to find the relevant table records.

Now I will provide the known information, which includes three types of data:
1. **Knowledge Graph Information:**
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.

2. **Image Data**: Presented in a Q&A format, including image data id, image title, image-based questions, and their answers.Each image title is an entity.

3. **Passages**: Includes text data id ,title and the text content.

{userful_information}

Next, I will provide the natural language question, table information, and some partial table data (the table data may not be fully provided, and only part of the data will be shown).
The table information contains the table's title (which is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, and a description of the table.
The internal table data format is as follows:
cell value: <cell value> 
corresponding column: <the name of the corresponding column>
(This means the cell value and the column name to which this value belongs.)

{table_information}
Question:{question}

        '''

        prompt += '''
Our task is to identify the relevant table records that can help answer the question. Your job is to analyze the given information, identify all the column names related to the question (as there may be more than one), and the filtering conditions, so as to find the relevant table records.
To complete this task:
First, determine whether the given table is relevant to the question based on the provided table information.If the table is not relevant, directly output 'not relevant', do not proceed with any further operations
If the table is relevant to the question, analyze the given information, identify all the column names related to the question (as there may be more than one), and the filtering conditions, so as to find the relevant table records,并且按照指定的json格式输出。

If the table is not relevant, directly output 'not relevant'(no any explanation)：
'not relevant'

If the table is relevant, multiple queries can be performed on a single table, output the result in the following JSON format(no any explanation, no ```json```):
{
    "QueryTable-1":{   
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or {
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    },
    "QueryTable-2":{   
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or { 
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    },
    ....
    "QueryTable-n":{
        "select columns": <the column names to search for, which are usually more than one to avoid missing important information, columns are separated by "@#">,
        "conditions": {} or { 
            <column value1>: <The column name to which the column value1 belongs>,
            <column value2>: <The column name to which the column value2 belongs>
            ...
        }
    }

}
        '''
        return prompt
#         _, table_information = self._get_information(table_entity_data_list)

#         prompt = f'''
# Now, I will provide the QueryTable API includes the API name, description, input, and output.
# API: QueryTable  
# Description:This API is designed for table-modal data. Before calling this API, input a list of filtering conditions (which may not always be required) and the columns you wish to search for. The output will be the specific column values corresponding to the searched columns from the records filtered based on the given conditions. This API will provide the table name, column names, and condition list in advance. If the table name is empty, this API cannot be called.
# The column names to be searched for in the input are selected from the provided table's column names based on the question. The columns you search for should contain the information you need to help answer the question.
# The filtering conditions in the input are selected from the given filtering conditions that you believe are helpful in answering the question.
# The provided table name, column names, and filtering conditions are as follows:
# {table_information}
# '''
#         prompt += '''
# Please determine the input for the QueryTable API based on the provided table information.
# Input: {
#     "select columns":<The select columns are not only one and the columns to search must be selected from the provided table columns. These columns contain the information needed to help answer the question(columns are separated by "@#")>
#     "conditions":{}or{
#         <column value1>:<The column name to which the column value1 belongs>,
#         <column value2>:<The column name to which the column value2 belongs>
#         ...
#     },
#  }
# Output: The column values corresponding to the searched column names. The format of the output is as follows:
# column name:select column1|select column2|...
# row 1:specific value of select column1|specific value of select column2|..
# row 2:...
# ''' 
#         prompt += '''
# Please output the API you will call. The output format is as follows:
# Output format:
# {
#     "API Access": {
#         "API": "<Interface name>",
#         "Input": <JSON-formatted input>
#     }
# }
#         '''
#         return prompt

