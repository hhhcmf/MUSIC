class KGReasonPrompt:
    def __init__(self, example=None):
        self.example = example
    
    def extract_useful_information(self, model_output_text):
        keywords = [
            "useful knowledge graph information",
            "useful passages data",
            "useful table data",
            "useful image data"
        ]

        result = {
            "useful_knowledge_graph_information": "",
            "useful_passages_data": "",
            "useful_table_data": "",
            "useful_image_data": ""
        }

        lines = model_output_text.splitlines()

        current_section = None
        section_start_index = None

        for index, line in enumerate(lines):
            line_lower = line.strip().lower()

            for keyword in keywords:
                if keyword.lower() in line_lower:
                    if current_section:
                        section_content = "\n".join(lines[section_start_index:index]).strip()
                        result[current_section] = section_content

                    current_section = keyword.replace(" ", "_")
                    section_start_index = index
                    break

        if current_section:
            section_content = "\n".join(lines[section_start_index:]).strip()
            result[current_section] = section_content

        return result
    
    def get_table_record_data_information(self, table_record_data):
        table_name, table_desp, select_columns, conditions, output = table_record_data

    def create_table_reason_prompt(self, question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag = 0):
        extract_useful_information = self.extract_useful_information(userful_information)
        prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
1. **Knowledge Graph Information:**
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.
2. **Image Data**: Presented in a Q&A format, including image data id, image title, image-based questions, and their answers.

3. **Passages**: Includes text data id ,title and the text content.

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{passage_information}
{image_qa_information}

**Task Requirements:**

1. **Analyze the Question:**
 - Understand and identify the conditions and requirements within the question.

2. **Extract Intermediate Answers from Table Data:**
 - Only extract relevant intermediate entities from the specific cell values in the table (excluding column names).
 - Intermediate answers should aid in advancing the reasoning process and should not directly provide the conditions stated in the question.
 - Exclude information that duplicates the question's conditions.

3. **Output Format:**
If there is no relevant content in the table:
output `no intermediate answer`.
else:
If there are relevant contents in the table.List the extracted intermediate entity names, each on a separate line:
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>


**Important Notes:**
- **Analyze All Known Data**: When extracting intermediate answers, you can reference information from the knowledge graph, image data, and text fragments to aid in understanding and reasoning.
- **Only Extract from Table Data's Cell Values**: The final intermediate answers must come from the table data's cell values and should not involve other data types.
- **Intermediate Answers Should Be Useful for Reasoning**: They should be generated during the reasoning process and not be conditions already provided in the question. 

Question:{question}
{table_record_information}

'no intermediate answer'或者
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>
        '''
        if gpt_4_flag:
            prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
1. **Knowledge Graph Information:**
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.
2. **Image Data**: Presented in a Q&A format, including image data id, image title, image-based questions, and their answers.

3. **Passages**: Includes text data id ,title and the text content.

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{passage_information}
{image_qa_information}

**Task Requirements:**
Analyze the given question and known knowledge, and extract the intermediate answer entities produced during the intermediate reasoning process from the reasoning results of the table below. Each cell value in the table is an entity.

Question:{question}
{table_record_information}

please extract the intermediate answer entities from the table
Please output in the following format. Note that there may be multiple intermediate answers, and if there are multiple, output one per line. Each intermediate answer does not need numbering or any embellishment.
'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>
            '''
        return prompt

    def create_text_reason_prompt(self, question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag = 0):
        extract_useful_information = self.extract_useful_information(userful_information)
        prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
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

3. **Passages**: Includes text data id ,title ,text and related triples.
The triple format is: (head entity, relation, tail entity): triples description.

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{image_qa_information}
{table_record_information}

**Task Requirements:**

1. **Analyze the Question:**
 - Understand and identify the conditions and requirements within the question.

2. **Extract Intermediate Answers from Passages Data:**
 - Only extract relevant intermediate entities from Passages Data.
 - Intermediate answers should aid in advancing the reasoning process and should not directly provide the conditions stated in the question.
 - Exclude information that duplicates the question's conditions.

3. **Output Format:**
If there is no relevant content in the passages:
output `no intermediate answer`.
else:
If there are relevant contents in the passages.List the extracted intermediate entity names, each on a separate line:
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>


**Important Notes:**
- **Analyze All Known Data**: When extracting intermediate answers, you can reference information from the knowledge graph, image data, and table data to aid in understanding and reasoning.
- **Only Extract from passages**: The final intermediate answers must come from the passages and should not involve other data types.
- **Intermediate Answers Should Be Useful for Reasoning**: They should be generated during the reasoning process and not be conditions already provided in the question. 

Question:{question}
{passage_information}

'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>
        '''

        prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
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

3. **Passages**: Includes text data id ,title ,text and related triples.
The triple format is: (head entity, relation, tail entity): triples description.

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{image_qa_information}
{table_record_information}

**Task Requirements:**
**Task Requirements:**
Analyze the given question and known knowledge, and extract the intermediate answer entities produced during the intermediate reasoning process from the reasoning results of the passages data below. 
Each passage title, the entities in the passage, and the head entity, tail entity of triples are entities.

Question:{question}
{passage_information}

please extract the intermediate answer entities from the passages data
Please output in the following format.Note that there may be multiple intermediate answers, and if there are multiple, output one per line. Each intermediate answer does not need numbering or any embellishment.
'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>
        '''
        return prompt
        

    def create_image_reason_prompt(self, question, userful_information, passage_information, image_qa_information, table_record_information, gpt_4_flag = 0):
        extract_useful_information = self.extract_useful_information(userful_information)
        prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
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

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{passage_information}
{table_record_information}

**Task Requirements:**
1. **Analyze the Question:**
 - Understand and identify the conditions and requirements within the question.
2. **Extract Intermediate Answers from Image Data:**
 - Only extract relevant intermediate entities from the titles in the image data .
 - Intermediate answers should aid in advancing the reasoning process and should not directly provide the conditions stated in the question.
 - Exclude information that duplicates the question's conditions.
3. **Output Format:**
If there is no relevant content in the image:
output `no intermediate answer`.
else:
If there are relevant contents in the image data.List the extracted intermediate entity names, each on a separate line:
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>

**Important Notes:**
- **Analyze All Known Data**: When extracting intermediate answers, you can reference information from the knowledge graph, table data, and text fragments to aid in understanding and reasoning.
- **Only Extract from Image Data's Titles**: The final intermediate answers must come from the image data's title and should not involve other data types.
- **Image data is not data that involves image entities in a given knowledge graph.
- **Intermediate Answers Should Be Useful for Reasoning**: They should be generated during the reasoning process and not be conditions already provided in the question. 

Question:{question}
{image_qa_information}

'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>

        '''
        if gpt_4_flag:
            prompt = f'''
You are a data analyst.

**Task Description:**

You will be provided with the following types of data:
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

4. **Table Data**: Contains multiple records, with each cell value being an entity.

**Known Data:**
{extract_useful_information['useful_knowledge_graph_information']}
{passage_information}
{table_record_information}

**Task Requirements:**
Analyze the given question and known knowledge, and extract the intermediate answer entities produced during the intermediate reasoning process from the reasoning results of the image data below. Each image title is an entity.

Question:{question}
{image_qa_information}

please extract the intermediate answer entities from the image.
Please output in the following format. Note that there may be multiple intermediate answers, and if there are multiple, output one per line. Each intermediate answer does not need numbering or any embellishment.
'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>

            '''

        return prompt

    def create_related_entity_prompt(self, question, entity_information):
        prompt = f'''
Given a natural language question and entity information in the knowledge graph, find all entities in the given entity information that are relevant to the question.
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.

Knowledge Graph Information:
{entity_information}

Now I will give the question:
Question:{question}

If there are no relevant entities, please output "no relevant entity".
Please find the entities related to the question and output in the following format:
"no relevant entity" or
entity name:<entity name> data id:<data id >
entity name:<entity name> data id:<data id >
...
entity name:<entity name> data id:<data id >
        '''
        return prompt
    
    def create_kg_reason_prompt(self, question, useful_information):
        prompt = f'''
You are a data analyst.

You will be provided knowledge graph information:
The knowledge graph contains entities in three modalities: image, text, and table.An entity may have multiple modalities, which would correspond to multiple modality information.
For image-modality entities, each entity's information contains the data id, the title of the image, related triples, the entity's data type and the information of alias entities.
Data type: image title or object, indicating whether the entity is the title of the image or an object within the image.
The triple format is: (image title, has, object).
For text-modalityentities, each entity's information contains the data id, a description of the entity, the text detail data(the content of the text block corresponding to this entity), related triples, a description of those triples and the information of alias entities.
The triple format is: (head entity, relation, tail entity): triples description.
For table-modality entities, each entity's information contains the data id, the data type, the table's title (the title is a descriptive expression of the table, not the table name or specific values within the table), the table name, column names, a description of the table and the information of alias entities.
Data type: column name, column value, table title, or table name, indicating in what form the entity appears within the table. If the data type is "column value", the entity will also include the column name that the value belongs to in the table.

Task Requirements:
Analyze the given question and knowledge graph information, and extract the intermediate answer entities produced during the intermediate reasoning process.

Question: {question}
Knowledge graph information:
{useful_information}

please extract the intermediate answer entities from the knowledge graph.
Please output in the following format. Note that there may be multiple intermediate answers, and if there are multiple, output one per line. Each intermediate answer does not need numbering or any embellishment.
'no intermediate answer' or
Intermediate answers:
<Intermediate Answers 1>
...
<Intermediate Answers n>
        '''
        return prompt
