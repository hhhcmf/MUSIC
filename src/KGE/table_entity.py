import os
import sqlite3

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
    table_name = table_data["table"].get("table_name", "")
    title = table_data.get("title", "")
    if title:
        knowledge_graph.append((table_name, "belongs_to", title))

    # 提取列名
    column_names = [col["column_name"] for col in columns]
    key_pos, null_cols = [], []
    for i in range(0, len(columns)):
        col = columns[i]
        if col['metadata'].get('is_key_column'):
            key_pos.append(i)
        if not col['column_name'] or col['column_name'] == '':
            null_cols.append(i)
        else:
            if table_name:
                knowledge_graph.append((col['column_name'], "table_name", table_name))



    # 遍历表格的每一行
    alias = {}
    for row in rows:
        head_entity = None
        for i, cell in enumerate(row):
            if i in null_cols:
                continue
            head_entity = ''
            # 处理头实体（假定 "Team" 列为头实体列）
            if i in key_pos:  # 假定第二列 "Team" 是头实体
                head_entity = cell['text']
                knowledge_graph.append((head_entity, "is", column_names[i]))
                links = cell.get('links')
                for link in links:
                    if link.get('wiki_title'):
                        if cell['text'] != link.get('wiki_title'):
                            alias[link.get('wiki_title')] = cell["text"]
                            alias[cell["text"]] = link.get('wiki_title')
                            knowledge_graph.append((cell["text"], "equivalent_to", link.get('wiki_title')))

            if head_entity:
                for j, cell in enumerate(row):
                    if j != i and j not in null_cols:
                        relation = column_names[j]
                        tail_entity = cell['text']  
                        knowledge_graph.append((head_entity, relation, tail_entity))

                        links = cell.get('links')
                        for link in links:
                            if link.get('wiki_title'):
                                if cell['text'] != link.get('wiki_title'):
                                    alias[link.get('wiki_title')] = cell["text"]
                                    alias[cell["text"]] = link.get('wiki_title')
                                    knowledge_graph.append((cell["text"], "equivalent_to", link.get('wiki_title')))

    return knowledge_graph

test_kg_path = '/home/cmf/multiQA/KG_LLM/KGE/data/kg.db'
    
def insert_relations(relation_list):
    connection = sqlite3.connect(test_kg_path)
    cursor = connection.cursor()
    for head_entity, relation, tail_entity in relation_list:
        cursor.execute("""
        INSERT INTO relationships (head_entity, relation, tail_entity) 
        VALUES (?, ?, ?)
        """, (head_entity.upper(), relation.upper(), tail_entity.upper()))
    connection.commit()
    print(f"Inserted {len(relation_list)} relations into 'relationships' table.")
    cursor.close()
    connection.close()


def save_knowledge_graph_to_txt(knowledge_graph, output_file):
    """
    将知识图谱和等价实体关系保存到 txt 文件中。
    :param knowledge_graph: 知识图谱的三元组列表。
    :param equivalences: 等价实体的三元组列表。
    :param output_file: 保存的文件路径。
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple in knowledge_graph:
            f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
            
        
    print(f"知识图谱已保存到 {output_file}")


# 示例表格数据
table_data = {
"title": "1998–99 Saudi Premier League", 
"url": "https://en.wikipedia.org/wiki/1998–99_Saudi_Premier_League", 
"id": "8cd195abf225852ec4d83fdf79622a88", 
"table": {
    "table_rows": [[{"text": "1", "links": []}, {"text": "Al-Hilal", "links": [{"text": "Al-Hilal", "wiki_title": "Al-Hilal FC", "url": "https://en.wikipedia.org/wiki/Al-Hilal_FC"}]}, {"text": "22", "links": []}, {"text": "15", "links": []}, {"text": "3", "links": []}, {"text": "4", "links": []}, {"text": "39", "links": []}, {"text": "22", "links": []}, {"text": "+17", "links": []}, {"text": "48", "links": []}], [{"text": "2", "links": []}, {"text": "Al-Ittihad", "links": [{"text": "Al-Ittihad", "wiki_title": "Al-Ittihad (Jeddah)", "url": "https://en.wikipedia.org/wiki/Al-Ittihad_(Jeddah)"}]}, {"text": "22", "links": []}, {"text": "15", "links": []}, {"text": "6", "links": []}, {"text": "1", "links": []}, {"text": "45", "links": []}, {"text": "32", "links": []}, {"text": "+13", "links": []}, {"text": "48", "links": []}], [{"text": "3", "links": []}, {"text": "Al-Ahli", "links": [{"text": "Al-Ahli", "wiki_title": "Al-Ahli (Jeddah)", "url": "https://en.wikipedia.org/wiki/Al-Ahli_(Jeddah)"}]}, {"text": "22", "links": []}, {"text": "10", "links": []}, {"text": "8", "links": []}, {"text": "4", "links": []}, {"text": "38", "links": []}, {"text": "22", "links": []}, {"text": "+16", "links": []}, {"text": "38", "links": []}], [{"text": "4", "links": []}, {"text": "Al-Shabab", "links": [{"text": "Al-Shabab", "wiki_title": "Al Shabab FC (Riyadh)", "url": "https://en.wikipedia.org/wiki/Al_Shabab_FC_(Riyadh)"}]}, {"text": "22", "links": []}, {"text": "10", "links": []}, {"text": "8", "links": []}, {"text": "4", "links": []}, {"text": "40", "links": []}, {"text": "23", "links": []}, {"text": "+17", "links": []}, {"text": "38", "links": []}], [{"text": "5", "links": []}, {"text": "Al-Nasr", "links": [{"text": "Al-Nasr", "wiki_title": "Al Nassr FC", "url": "https://en.wikipedia.org/wiki/Al_Nassr_FC"}]}, {"text": "22", "links": []}, {"text": "10", "links": []}, {"text": "3", "links": []}, {"text": "9", "links": []}, {"text": "33", "links": []}, {"text": "24", "links": []}, {"text": "+9", "links": []}, {"text": "33", "links": []}], [{"text": "6", "links": []}, {"text": "Al-Ettifaq", "links": [{"text": "Al-Ettifaq", "wiki_title": "Ettifaq FC", "url": "https://en.wikipedia.org/wiki/Ettifaq_FC"}]}, {"text": "22", "links": []}, {"text": "8", "links": []}, {"text": "6", "links": []}, {"text": "8", "links": []}, {"text": "29", "links": []}, {"text": "30", "links": []}, {"text": "-1", "links": []}, {"text": "30", "links": []}], [{"text": "7", "links": []}, {"text": "Al-Riyadh", "links": [{"text": "Al-Riyadh", "wiki_title": "Al-Riyadh", "url": "https://en.wikipedia.org/wiki/Al-Riyadh"}]}, {"text": "22", "links": []}, {"text": "7", "links": []}, {"text": "6", "links": []}, {"text": "9", "links": []}, {"text": "30", "links": []}, {"text": "32", "links": []}, {"text": "-2", "links": []}, {"text": "27", "links": []}], [{"text": "8", "links": []}, {"text": "Al-Ta'ee", "links": [{"text": "Al-Ta'ee", "wiki_title": "Al-Ta'ee", "url": "https://en.wikipedia.org/wiki/Al-Ta'ee"}]}, {"text": "22", "links": []}, {"text": "7", "links": []}, {"text": "4", "links": []}, {"text": "11", "links": []}, {"text": "24", "links": []}, {"text": "40", "links": []}, {"text": "-16", "links": []}, {"text": "25", "links": []}], [{"text": "9", "links": []}, {"text": "Al Nejmeh", "links": [{"text": "Al Nejmeh", "wiki_title": "Al-Najma (Saudi Arabian Sport Club)", "url": "https://en.wikipedia.org/wiki/Al-Najma_(Saudi_Arabian_Sport_Club)"}]}, {"text": "22", "links": []}, {"text": "6", "links": []}, {"text": "5", "links": []}, {"text": "11", "links": []}, {"text": "26", "links": []}, {"text": "33", "links": []}, {"text": "-23", "links": []}, {"text": "23", "links": []}], [{"text": "10", "links": []}, {"text": "Al Wahda", "links": [{"text": "Al Wahda", "wiki_title": "Al-Wehda Club (Mecca)", "url": "https://en.wikipedia.org/wiki/Al-Wehda_Club_(Mecca)"}]}, {"text": "22", "links": []}, {"text": "6", "links": []}, {"text": "3", "links": []}, {"text": "13", "links": []}, {"text": "32", "links": []}, {"text": "46", "links": []}, {"text": "-14", "links": []}, {"text": "21", "links": []}], [{"text": "11", "links": []}, {"text": "Al-Ansar", "links": [{"text": "Al-Ansar", "wiki_title": "Al-Ansar (Saudi Arabia)", "url": "https://en.wikipedia.org/wiki/Al-Ansar_(Saudi_Arabia)"}]}, {"text": "22", "links": []}, {"text": "5", "links": []}, {"text": "6", "links": []}, {"text": "11", "links": []}, {"text": "20", "links": []}, {"text": "34", "links": []}, {"text": "-14", "links": []}, {"text": "21", "links": []}], [{"text": "12", "links": []}, {"text": "Hajer", "links": [{"text": "Hajer", "wiki_title": "Hajer Club", "url": "https://en.wikipedia.org/wiki/Hajer_Club"}]}, {"text": "22", "links": []}, {"text": "2", "links": []}, {"text": "7", "links": []}, {"text": "13", "links": []}, {"text": "21", "links": []}, {"text": "39", "links": []}, {"text": "-18", "links": []}, {"text": "13", "links": []}]], 
    "table_name": "Final league table", 
    "header": [
        {"column_name": "", "metadata": {"parsed_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], "is_index_column": True, "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "Team", "metadata": {"num_of_links": 12, "ner_appearances_map": {"PERSON": 10}, "is_key_column": True, "image_associated_column": True, "entities_column": True}}, 
        {"column_name": "GP", "metadata": {"parsed_values": [22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "W", "metadata": {"parsed_values": [15.0, 15.0, 10.0, 10.0, 10.0, 8.0, 7.0, 7.0, 6.0, 6.0, 5.0, 2.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "D", "metadata": {"parsed_values": [3.0, 6.0, 8.0, 8.0, 3.0, 6.0, 6.0, 4.0, 5.0, 3.0, 6.0, 7.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "L", "metadata": {"parsed_values": [4.0, 1.0, 4.0, 4.0, 9.0, 8.0, 9.0, 11.0, 11.0, 13.0, 11.0, 13.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "GF", "metadata": {"parsed_values": [39.0, 45.0, 38.0, 40.0, 33.0, 29.0, 30.0, 24.0, 26.0, 32.0, 20.0, 21.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "GA", "metadata": {"parsed_values": [22.0, 32.0, 22.0, 23.0, 24.0, 30.0, 32.0, 40.0, 33.0, 46.0, 34.0, 39.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}, 
        {"column_name": "GD", "metadata": {"parsed_values": [17.0, 13.0, 16.0, 17.0, 9.0, -1.0, -2.0, -16.0, -23.0, -14.0, -14.0, -18.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {}}}, 
        {"column_name": "Pts", "metadata": {"parsed_values": [48.0, 48.0, 38.0, 38.0, 33.0, 30.0, 27.0, 25.0, 23.0, 21.0, 21.0, 13.0], "type": "float", "num_of_links": 0, "ner_appearances_map": {"CARDINAL": 12}}}]}
}


# 生成完整知识图谱和等价实体
knowledge_graph= generate_complete_knowledge_graph_with_equivalences(table_data)
insert_relations(knowledge_graph)
# 保存到 txt 文件
# output_file = "/home/cmf/multiQA/KG_LLM/KGE/data/knowledge_graph.txt"
# save_knowledge_graph_to_txt(knowledge_graph, output_file)
