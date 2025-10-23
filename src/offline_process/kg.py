import json
import sqlite3
from data.MMQA.preprocess import get_dev_information, kg_path
import pandas as pd
#TODO add entity data id to table relationships
class Node:
    def __init__(self, entity_id, entity_name, entity_type, description, title, text = '', data_id='', data_type='', isColumn=0, isColumnValue=0, table_name='', columns=[], column='', isImageTitle=0, wiki_title ='', url ='', path = None):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.description = description
        self.title = title
        self.text = text
        self.data_id = data_id
        self.type = data_type
        self.is_column =isColumn
        self.is_column_value = isColumnValue
        self.table_name = table_name
        self.columns = columns
        self.column = column
        self.isImageTitle = isImageTitle
        self.wiki_title = wiki_title
        self.url = url
        self.image_path = path
        self.related_snippet = ''
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.entity_name == other.entity_name and
                    self.entity_type == other.entity_type and
                    self.description == other.description and
                    self.title == other.title and
                    self.text == other.text and
                    self.data_id == other.data_id and
                    self.type == other.type and
                    self.is_column == other.is_column and
                    self.is_column_value == other.is_column_value and
                    self.table_name == other.table_name and
                    self.columns == other.columns and
                    self.column == other.column and
                    self.isImageTitle == other.isImageTitle and
                    self.wiki_title == other.wiki_title and
                    self.url == other.url)
        return False


    # 定义 __hash__ 方法，确保对象可以在 set 和 dict 中使用
    def __hash__(self):
        return hash((self.entity_name, self.entity_type, self.description, self.title, self.text, 
                     self.data_id, self.type, self.is_column, self.is_column_value, self.table_name, 
                     self.columns, self.column, self.isImageTitle, self.wiki_title, self.url))
    def __str__(self):
        return f"Node({self.entity_name}, {self.title}, {self.entity_id}, {self.type}, {self.data_id}, {self.description})"

class Edge:
    def __init__(self, relation_id, data_id, head_entity, tail_entity, relation, description,  text = '', data_type = '', strength = ''):
        self.relation_id = relation_id
        self.data_id = data_id
        self.head_entity = head_entity
        self.tail_entity = tail_entity
        self.relation = relation
        self.description = description
        self.text = text
        self.type = data_type
        self.strength = strength

    def __str__(self):
        return f"Edge({self.head_entity} -> {self.tail_entity}, {self.relation}, {self.description})"
    
    def __repr__(self):
        return self.__str__()

class KnowledgeGraph:
    def __init__(self, db_path):
        self.db_path = db_path
       
        self.entity_table = 'entities'
        self.relation_table = 'relationships'
    
    def _get_connection(self):
     
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _enable_wal_mode(self):
        with self._get_connection() as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
    

    def close(self):
        # 无需额外的关闭方法，因为使用上下文管理器会自动关闭连接
        pass
    def create_kg_temp_tables(self, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])

        # 使用一个连接完成所有操作
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 删除已有的临时表，如果存在
            drop_temp_tables_sql = '''
                DROP TABLE IF EXISTS entities_view;
                DROP TABLE IF EXISTS relationships_view;
            '''
            cursor.executescript(drop_temp_tables_sql)

            # 创建临时实体表
            create_temp_entities_sql = f'''
                CREATE TEMP TABLE entities_view AS
                SELECT data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url
                FROM {self.entity_table}
                WHERE data_id IN ({data_id_list_str})
            '''
            cursor.execute(create_temp_entities_sql)

            # 创建临时关系表
            create_temp_relationships_sql = f'''
                CREATE TEMP TABLE relationships_view AS
                SELECT data_id, head_entity, tail_entity, relation, description, text, type
                FROM {self.relation_table}
                WHERE data_id IN ({data_id_list_str})
            '''
            cursor.execute(create_temp_relationships_sql)

            # 确保更改被提交
            conn.commit()
            
    def create_kg_view(self, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            drop_view_sql = "DROP VIEW IF EXISTS entities_view"
            cursor.execute(drop_view_sql)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 创建新的视图
            sql = f'''
                CREATE VIEW entities_view AS
                SELECT data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path
                FROM {self.entity_table}
                WHERE data_id IN ({data_id_list_str})
            '''
            cursor.execute(sql)
        with self._get_connection() as conn:
            drop_view_sql = "DROP VIEW IF EXISTS relationships_view"
            cursor.execute(drop_view_sql)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = f'''
                CREATE VIEW relationships_view AS
                SELECT data_id, head_entity, tail_entity, relation, description, text, type
                FROM {self.relation_table}
                WHERE data_id IN ({data_id_list_str})
            '''
            cursor.execute(sql)

    def drop_kg_view(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = 'DROP VIEW IF EXISTS entities_view'
            cursor.execute(sql)
            sql = 'DROP VIEW IF EXISTS relationships_view'
            cursor.execute(sql)

    def get_entity_by_entityId(self, entity_id):
        entity_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # print(type(entity_id))
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities WHERE id = ?"
            cursor.execute(query, (int(entity_id),))
            results = cursor.fetchall()
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list[0]

    def get_entity_by_entity(self, entity_name, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        entity_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities WHERE entity = ? and data_id IN ({data_id_list_str})"
            cursor.execute(query, (entity_name,))
            results = cursor.fetchall()
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        entity_dict = {}  # 键为 (entity_name, data_id)，值为 Node 对象
        for entity in entity_list:
            key = (entity.entity_name, entity.data_id)
            if key not in entity_dict:
                entity_dict[key] = entity
            else:
                # 比较现有实体与当前实体的 description 长度，保留更长的
                current_desc_len = len(entity_dict[key].description or "")
                new_desc_len = len(entity.description or "")
                if new_desc_len > current_desc_len:
                    entity_dict[key] = entity
        
        # 将字典的值转为列表并返回
        return list(entity_dict.values())
        
    def get_title_by_data_id(self, data_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT  title from entities WHERE data_id = ?"
            cursor.execute(query, (str(data_id),))
            result = cursor.fetchone()
        return result[0]
        
    
    def get_related_entity_name_by_entity(self, entity_name, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        entity_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, head_entity, tail_entity, relation FROM relationships WHERE (head_entity = ? or tail_entity= ?)"
            cursor.execute(query, (entity_name, entity_name))
            results = cursor.fetchall()
            for relation_id, head_entity, tail_entity, relation in results:
                entity_list.append(head_entity)
                entity_list.append(tail_entity)
        return entity_list
    
    def get_entity_by_entityDesp(self, entity_desp, data_id_list):
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities WHERE description like ? and data_id IN ({data_id_list_str})"
            entity_desp = f'%{entity_desp}%'
            cursor.execute(query, (entity_desp,))
            results = cursor.fetchall()
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list
        
    def get_kg_entities(self, data_id_list):
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where entity!='' and data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            # PRINT(len(results))
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list

    def get_kg_entity_names(self, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        entity_name_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT entity FROM entities where data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            entity_name_list = [entity[0] for entity in results if entity[0] != '']
        return entity_name_list
    
    
    def get_image_entity_data(self, title, data_id_list):
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where entity=? and title = ? and type='image' and data_id IN ({data_id_list_str})"
            cursor.execute(query,(title, title))
            results = cursor.fetchall()
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list[0]

    def get_entity_by_title(self, title, modality_type, data_id_list):
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where title={title} and type={modality_type} and data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            # PRINT(len(results))
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list[0]
    
    def get_entity_by_table_name(self, title, data_id_list):
        # print('table name:', title)
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where table_name=? and data_id IN ({data_id_list_str})"
            cursor.execute(query,(title,))
            results = cursor.fetchall()
            # PRINT(len(results))
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list[0]

    def get_entity_by_IdAndEntity(self, data_id, entity):
        entity_data = None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where data_id = ? and entity = ?"
            cursor.execute(query, (data_id, entity))
            results = cursor.fetchall()
            # print(data_id, entity)
            # print(results)
            if results:
                entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path =  results[0]
                entity_data = Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path)
        return entity_data
    
    def get_entity_by_entityAndType(self, entity, d_type, data_id_list):
        entity_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, entity, entity_type, type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path FROM entities where entity=? and type=? and data_id IN ({data_id_list_str})"
            cursor.execute(query,(entity, d_type))
            results = cursor.fetchall()
            # PRINT(len(results))
            for entity_id, data_id, entity_name, entity_type, data_type, title, description, text, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path in results:
                entity_list.append(Node(entity_id, entity_name, entity_type, description, title, text, data_id, data_type, isColumn, isColumnValue, table_name, columns, column, isImageTitle, wiki_title, url, image_path))
        return entity_list
    
    def get_relation_by_id(self, relation_id):
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, head_entity, tail_entity, relation, description, text, type, strength FROM relationships WHERE id = ?"
            cursor.execute(query, (int(relation_id),))
            results = cursor.fetchall()
            for relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength in results:
                relation_list.append(Edge(relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength))
        return relation_list[0]

    def get_relation_by_entity(self, entity_name, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type,strength  FROM relationships WHERE (head_entity = ? or tail_entity= ?) and data_id IN ({data_id_list_str})"
            cursor.execute(query, (entity_name, entity_name))
            results = cursor.fetchall()
        return results

    def get_triples(self, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type  FROM relationships where data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            for data_id, head_entity, tail_entity, relation, description, text, data_type in results:
                relation_list.append((head_entity, relation, tail_entity))
        return relation_list

    def get_kg_relationships(self, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT data_id, head_entity, tail_entity, relation, description, text, type, strength  FROM relationships where data_id IN ({data_id_list_str})"
            cursor.execute(query)
            results = cursor.fetchall()
            for data_id, head_entity, tail_entity, relation, description, text, data_type, strength in results:
                relation_list.append(Edge(data_id, head_entity, tail_entity, relation, description, text, data_type, strength))
        return relation_list

    def get_relation_by_triple(self, head_entity, relation, tail_entity, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, head_entity, tail_entity, relation, description, text, type, strength FROM relationships WHERE head_entity = ? and relation = ? and tail_entity= ? and data_id IN ({data_id_list_str})"
            cursor.execute(query, (head_entity, relation, tail_entity))
            results = cursor.fetchall()
            for relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength in results:
                relation_list.append(Edge(relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength))
        return relation_list

    def get_triples_by_entity(self, entity_name, data_id):
        # data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, head_entity, tail_entity, relation, description, text, type, strength FROM relationships WHERE (head_entity = ? or tail_entity= ?) and data_id = ? and type == 'text'"
            cursor.execute(query, (entity_name, entity_name, str(data_id)))
            results = cursor.fetchall()
            for relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength in results:
                relation_list.append(Edge(relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength))
        return relation_list

    def get_triples_by_data_id(self, data_id):
        relation_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id, data_id, head_entity, tail_entity, relation, description, text, type, strength FROM relationships WHERE data_id = ?"
            cursor.execute(query, (str(data_id),))
            results = cursor.fetchall()
            for relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength in results:
                relation_list.append(Edge(relation_id, data_id, head_entity, tail_entity, relation, description, text, data_type, strength))
        return relation_list
    
    def get_image_path(self, image_title, data_id_list):
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT image_path FROM entities where type = 'image' and title = ? and data_id IN ({data_id_list_str})"
            cursor.execute(query, (image_title,))
            result = cursor.fetchone()
        if result:
            image_path = result[0]
            return image_path
        return None
    
    def get_image_titles(self, data_id_list):
        image_title_list = []
        data_id_list_str = ', '.join([f"'{str(data_id)}'" for data_id in data_id_list])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT distinct(title) FROM entities where type='image' and data_id IN ({data_id_list_str})"
            cursor.execute(query)
            result = cursor.fetchall()
            image_title_list = [title[0] for title in result]
        return image_title_list
    
    def create_entity_link(self, kg_id):
        # print(self.db_path)
        # self.drop_entity_links()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 创建表 (如表已存在，此操作是幂等的，不会重复创建)
            try:
                # sql = 'DROP TABLE IF EXISTS entity_links_new'       
                # cursor.execute(sql)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS entity_links_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_id TEXT,
                    entity_name TEXT,
                    link_data_id TEXT,
                    link_entity_name TEXT,
                    type TEXT,
                    link_type TEXT,
                    kg_id INTEGER
                );
                ''')
                conn.commit()  # 提交表的创建
                print("Table entity_links_new created or already exists")
            except Exception as e:
                print(f"Error creating table: {e}")
                return
            cursor.execute('select * from entity_links_new')
            result = cursor.fetchall()
            # print(len(result))
            # 2. 获取一次性查询表中的数据
            cursor.execute("SELECT data_id, entity, wiki_title, url FROM entities_view WHERE type='table' AND wiki_title != ''")
            table_results = cursor.fetchall()

        # 3. 准备批量插入的数据（外部处理）
        insert_records = []
        entity_url_map = {}
        
        # 遍历查询结果，进行计算
        for data_id, entity, wiki_title, url in table_results:
            if (wiki_title, url) not in entity_url_map:
                # 这里不需要每次打开新连接来执行查询，只需用连接读取数据
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT data_id, entity, type FROM entities_view WHERE entity = ? and url = ? AND data_id != ? AND type != 'table'",
                        (wiki_title, url, data_id)
                    )
                    entity_url_map[(wiki_title, url)] = cursor.fetchall()

            # 获取到的结果
            results = entity_url_map[(wiki_title, url)]
            
            for link_data_id, link_entity, link_type in results:
                # 避免重复插入
                if (data_id, link_data_id) not in insert_records and (link_data_id, data_id) not in insert_records:
                    insert_records.append((data_id, entity, link_data_id, link_entity, kg_id, 'table', link_type))
                    insert_records.append((link_data_id, link_entity, data_id, entity, kg_id, link_type, 'table'))
        
        # print(table_results)
        # print(insert_records)

        # if insert_records:
        #     with self._get_connection() as conn:
        #         cursor = conn.cursor()
        #         insert_query = '''
        #         INSERT INTO entity_links_new (data_id, entity_name, link_data_id, link_entity_name, kg_id, type, link_type) 
        #         VALUES (?, ?, ?, ?, ?, ?, ?)
        #         '''
        #         cursor.executemany(insert_query, insert_records)
        #         conn.commit()  # 提交所有的插入操作

        # 4. 批量插入数据（重新打开连接进行插入操作）
        if insert_records:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                insert_query = '''
                INSERT INTO entity_links_new (data_id, entity_name, link_data_id, link_entity_name, kg_id, type, link_type) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.executemany(insert_query, insert_records)
                conn.commit()  # 提交所有的插入操作

        # 5. 处理重复数据（重新打开连接进行处理）
        # with self._get_connection() as conn:
        #     # 使用 pandas 进行重复处理，这部分可以放在数据库事务之外处理
        #     entities = pd.read_sql_query("SELECT entity, url, data_id, type FROM entities_view WHERE entity IS NOT NULL AND entity != '' AND entity != '-'", conn)

        # # 6. 找出重复的实体对并处理
        # duplicates = entities[entities.duplicated(subset=['url'], keep=False)]
        # # print(duplicates)

        # # 自连接查找重复实体对
        # merged_duplicates = duplicates.merge(
        #     duplicates, 
        #     on=['url'],  # 匹配条件是 entity 和 url
        #     suffixes=('_1', '_2')  # 区分两个表中的列名
        # )
        
        # # 过滤掉相同的记录
        # filtered_duplicates = merged_duplicates[merged_duplicates['data_id_1'] < merged_duplicates['data_id_2']]

        # # 提取需要的列
        # print(filtered_duplicates.columns)

        # result = filtered_duplicates.rename(
        #     columns={
        #         'entity_1': 'entity_name',
        #         'entity_2':'link_entity_name',
        #         'data_id_1': 'data_id',
        #         'data_id_2': 'link_data_id',
        #         'type_1': 'type',      # 对应第一个实体的 type
        #         'type_2': 'link_type'  # 对应第二个实体的 type
        #     }
        # )
        # # result['link_entity_name'] = result['entity_name'] 
        # result = result[['entity_name', 'data_id', 'link_entity_name', 'link_data_id', 'type', 'link_type']]

        # # 7. 写入临时表并进行插入操作
        # with self._get_connection() as conn:
        #     result.to_sql('entity_links_temp', conn, if_exists='replace', index=False)
        #     cursor = conn.cursor()
            
        #     # 插入符合条件的数据（过滤掉 url 和 entity 相同且 type 都是 'image' 的记录）
        #     cursor.execute(f'''
        #     INSERT INTO entity_links_new (entity_name, data_id, link_entity_name, link_data_id, kg_id, type, link_type)
        #     SELECT entity_name, data_id, link_entity_name, link_data_id, {kg_id}, type, link_type
        #     FROM entity_links_temp
        #     WHERE NOT (type = 'image' AND link_type = 'image')
        #     ''')

        #     # 第二次插入，反向插入数据
        #     cursor.execute(f'''
        #     INSERT INTO entity_links_new (entity_name, data_id, link_entity_name, link_data_id, kg_id, type, link_type)
        #     SELECT link_entity_name, link_data_id, entity_name, data_id, {kg_id}, link_type, type
        #     FROM entity_links_temp
        #     WHERE NOT (type = 'image' AND link_type = 'image')
        #     ''')

        #     conn.commit()  # 提交链接插入
    
    def create_entity_text_link(self, kg_id):
        print(self.db_path)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 创建表 (如表已存在，此操作是幂等的，不会重复创建)
            try:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS entity_links_new_1 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_id TEXT,
                    entity_name TEXT,
                    link_data_id TEXT,
                    link_entity_name TEXT,
                    type TEXT,
                    link_type TEXT,
                    kg_id INTEGER
                );
                ''')
                conn.commit()  # 提交表的创建
                print("Table entity_links_new_1 created or already exists")
            except Exception as e:
                print(f"Error creating table: {e}")
                return
            cursor.execute('select * from entity_links_new_1')
            result = cursor.fetchall()
            print(len(result))
            # 2. 获取一次性查询表中的数据
            cursor.execute("SELECT data_id, entity, wiki_title, url FROM entities_view WHERE type='table' AND wiki_title != ''")
            table_results = cursor.fetchall()

        # 3. 准备批量插入的数据（外部处理）
        insert_records = []
        entity_url_map = {}
        
        # 遍历查询结果，进行计算
        for data_id, entity, wiki_title, url in table_results:
            if (wiki_title, url) not in entity_url_map:
                # 这里不需要每次打开新连接来执行查询，只需用连接读取数据
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT data_id, entity, type FROM entities_view WHERE entity = ? AND url = ? AND data_id != ? AND type != 'table'",
                        (wiki_title, url, data_id)
                    )
                    entity_url_map[(wiki_title, url)] = cursor.fetchall()

            # 获取到的结果
            results = entity_url_map[(wiki_title, url)]
            
            for link_data_id, link_entity, link_type in results:
                # 避免重复插入
                if (data_id, link_data_id) not in insert_records and (link_data_id, data_id) not in insert_records:
                    insert_records.append((data_id, entity, link_data_id, link_entity, kg_id, 'table', link_type))
                    insert_records.append((link_data_id, link_entity, data_id, entity, kg_id, link_type, 'table'))
        
        # print(table_results)
        # print(insert_records)

        # 4. 批量插入数据（重新打开连接进行插入操作）
        # if insert_records:
        #     with self._get_connection() as conn:
        #         cursor = conn.cursor()
        #         insert_query = '''
        #         INSERT INTO entity_links_new_1 (data_id, entity_name, link_data_id, link_entity_name, kg_id, type, link_type) 
        #         VALUES (?, ?, ?, ?, ?, ?, ?)
        #         '''
        #         cursor.executemany(insert_query, insert_records)
        #         conn.commit()  # 提交所有的插入操作

        # 5. 处理重复数据（重新打开连接进行处理）
        with self._get_connection() as conn:
            # 使用 pandas 进行重复处理，这部分可以放在数据库事务之外处理
            entities = pd.read_sql_query("SELECT entity, url, data_id, type FROM entities_view WHERE entity IS NOT NULL AND entity != '' AND entity != '-'", conn)

        # 6. 找出重复的实体对并处理
        duplicates = entities[entities.duplicated(subset=['entity', 'url'], keep=False)]

        # 自连接查找重复实体对
        merged_duplicates = duplicates.merge(
            duplicates, 
            on=['entity', 'url'],  # 匹配条件是 entity 和 url
            suffixes=('_1', '_2')  # 区分两个表中的列名
        )
        
        # 过滤掉相同的记录
        filtered_duplicates = merged_duplicates[merged_duplicates['data_id_1'] < merged_duplicates['data_id_2']]

        # 提取需要的列
        result = filtered_duplicates.rename(
            columns={
                'entity': 'entity_name',
                'data_id_1': 'data_id',
                'data_id_2': 'link_data_id',
                'type_1': 'type',      # 对应第一个实体的 type
                'type_2': 'link_type'  # 对应第二个实体的 type
            }
        )
        result['link_entity_name'] = result['entity_name'] 
        result = result[['entity_name', 'data_id', 'link_entity_name', 'link_data_id', 'type', 'link_type']]

        # 7. 写入临时表并进行插入操作
        with self._get_connection() as conn:
            result.to_sql('entity_links_temp', conn, if_exists='replace', index=False)
            cursor = conn.cursor()
            
            # 插入符合条件的数据（过滤掉 url 和 entity 相同且 type 都是 'image' 的记录）
            cursor.execute(f'''
            INSERT INTO entity_links_new_1 (entity_name, data_id, link_entity_name, link_data_id, kg_id, type, link_type)
            SELECT entity_name, data_id, link_entity_name, link_data_id, {kg_id}, type, link_type
            FROM entity_links_temp
            WHERE NOT (type = 'image' AND link_type = 'image')
            ''')

            # 第二次插入，反向插入数据
            cursor.execute(f'''
            INSERT INTO entity_links_new_1 (entity_name, data_id, link_entity_name, link_data_id, kg_id, type, link_type)
            SELECT link_entity_name, link_data_id, entity_name, data_id, {kg_id}, link_type, type
            FROM entity_links_temp
            WHERE NOT (type = 'image' AND link_type = 'image')
            ''')

            conn.commit()  # 提交链接插入
            
    
    def remove_entity_link_duplicates(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 删除重复记录，只保留最小的 ROWID
            delete_query = """
            DELETE FROM entity_links_new_1
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM entity_links_new_1
                GROUP BY data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id
            );
            """
            cursor.execute(delete_query)
            conn.commit()
            print("Duplicate records removed successfully.")

        except Exception as e:
            print(f"SQLite error: {e}")

        finally:
            conn.close()
            
    def delete_entity_link(self):    
        conn = sqlite3.connect(self.db_path)
        print(self.db_path)
        try:
            cursor = conn.cursor()
            # 查询语句：从源数据库中获取 type 和 link_type 为 'text' 的记录
            select_sql = """
                SELECT id, data_id, entity_name, link_data_id, link_entity_name, type, link_type, kg_id
                FROM entity_links_new_1
                WHERE (type = 'text' and link_type = 'image') or  (type = 'image' and link_type = 'text');
            """
            
            # 执行查询
            cursor.execute(select_sql)

            # 获取所有符合条件的记录
            rows = cursor.fetchall()

            if rows:
                for idx_id, data_id, entity_name, link_data_id, link_entity_name, d_type, link_type, kg_id in rows:
                    title = self.get_title_by_data_id(data_id)
                    link_title = self.get_title_by_data_id(link_data_id)
                    # print(title, link_title)
                    if (entity_name != title and d_type == 'image') or (link_entity_name != link_title and link_type == 'image'):
                        print(f'---delete {idx_id}-----')
                        delete_sql = f'DELETE FROM entity_links_new_1 where id = {idx_id}'
                        cursor.execute(delete_sql)
                        conn.commit()
        except Exception as e:
            print(f"Error during database operations: {e}")
        finally:
            # 关闭游标和数据库连接
            cursor.close()
            conn.close()

        

    def get_entity_link(self, kg_id):
        entity_link_hash = {}
        record_list = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('select entity_name, data_id, link_entity_name, link_data_id from entity_links where kg_id = ?', (kg_id, ))
            results = cursor.fetchall()
        for entity_name, data_id, link_entity_name, link_data_id in results:
            entity_link_hash[(entity_name, data_id)] = []
        for entity_name, data_id, link_entity_name, link_data_id in results:
            if (entity_name, data_id, link_entity_name, link_data_id) not in record_list:
                record_list.append((entity_name, data_id, link_entity_name, link_data_id))
                entity_data = self.get_entity_by_IdAndEntity(link_data_id, link_entity_name)
                entity_link_hash[(entity_name, data_id)].append(entity_data)

        return entity_link_hash

    def drop_entity_links(self):        
        with self._get_connection() as conn:  # 确保在操作完成后自动关闭连接
            cursor = conn.cursor()
            
            # 删除表 entity_links
            # sql = 'DROP TABLE IF EXISTS entity_links'       
            # cursor.execute(sql)

            sql = 'DROP TABLE IF EXISTS entity_links_new_1'       
            cursor.execute(sql)
            
            # 删除视图 entity_links_temp
            sql = 'DROP TABLE IF EXISTS entity_links_temp'       
            cursor.execute(sql)

            # sql = 'DROP VIEW IF EXISTS entity_links_temp'       
            # cursor.execute(sql)
            
            # 提交事务
            conn.commit()


def create_indexes(db_path):
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 定义索引创建语句
    index_statements = [
        "CREATE INDEX IF NOT EXISTS data_id_index ON entities (data_id);",
        "CREATE INDEX IF NOT EXISTS data_id_index ON relationships (data_id);",
        "CREATE INDEX IF NOT EXISTS relation_data_id_index ON relationships (data_id);",
        "CREATE INDEX IF NOT EXISTS entity_index ON entities (entity);",
        "CREATE INDEX IF NOT EXISTS dentity_index ON entities (entity);", 
        "CREATE INDEX IF NOT EXISTS title_index ON entities (title);",
        "CREATE INDEX IF NOT EXISTS type_index ON entities (type);",
        "CREATE INDEX IF NOT EXISTS relation_type_index ON relationships (type);",
        "CREATE INDEX IF NOT EXISTS entity_data_id_index ON entities (entity, data_id);",
        "CREATE INDEX IF NOT EXISTS type_data_id_index ON entities (type, data_id);",
        "CREATE INDEX IF NOT EXISTS title_data_id_index ON entities (title, data_id);",
        "CREATE INDEX IF NOT EXISTS title_type_data_id_index ON entities (title, type, data_id);"
    ]

    try:
        for statement in index_statements:
            cursor.execute(statement)
            print(f"执行成功: {statement}")
    except sqlite3.OperationalError as e:
        print(f"执行失败: {e}")
    finally:
        # 提交更改并关闭连接
        conn.commit()
        conn.close()