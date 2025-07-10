import sqlite3
from utils.proprecss_llm_answer import extract_reasoning, fix_conditions
from queue import Queue
class SQLiteConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._init_pool()

    def _init_pool(self):
        """初始化连接池"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path)
            self._pool.put(conn)

    def get_connection(self):
        """从连接池获取连接，如果池空则尝试扩展池"""
        if self._pool.empty() and self.pool_size < self.max_pool_size:
            # 如果池空并且未达到最大连接数，扩展池
            conn = sqlite3.connect(self.db_path)
            self._pool.put(conn)
            self.pool_size += 1
            return conn
        elif not self._pool.empty():
            return self._pool.get()
        else:
            raise Exception("Connection pool exhausted and cannot expand further.")

    def release_connection(self, conn):
        """释放连接，返回池中"""
        if not self._pool.full():
            self._pool.put(conn)
        else:
            conn.close()  # 连接池满时，关闭连接

    def close_all(self):
        """关闭池中的所有连接"""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()
            
class TableProcessor:
    def __init__(self, base_config):
        self.base_config = base_config
        if base_config.args.dataset.lower() == 'mmqa':
            self.kg_path = '/data/xxx/mmqa/kg/kg.db'
            self.db_path = '/data/xxx/mmqa/MMQA.db'
        elif base_config.args.dataset.lower() == 'manymodalqa':
            self.kg_path = '/data/xxx/manymodalqa/kg/kg.db'
            self.db_path = '/data/xxx/manymodalqa/ManymodalQA.db'
        elif base_config.args.dataset.lower() == 'webqa':
            self.kg_path = ''
            self.db_path = ''
        else:
            raise f"The TableProcessor doesn't support {base_config.args.dataset}"
        
    def get_tableId_by_name(self, table_name):
        '''
        根据sql中的table name映射到id，以便后续得到真正的tablename（table_<table_id>）和col(col<id>_<table_id>)
        '''
        if self.base_config.args.dataset.lower() == 'mmqa':
            # print(self.kg_path)
            conn = sqlite3.connect(self.kg_path)
            cursor = conn.cursor()
            sql = 'select table_value from table_hash where table_key = ?'
            cursor.execute(sql, (table_name,))
            table_row = cursor.fetchone()

            if table_row is not None:
                table_id = table_row[0]
            else:
                table_id = None  

            cursor.close()
            conn.close()
            return table_id
        else:
            return table_id

    def get_cols_by_table_id(self, table_id):
        conn = sqlite3.connect(self.kg_path)
        cursor = conn.cursor()
        sql = 'select columns from entities where data_id = ?'
        cursor.execute(sql, (table_id,))
        columns = cursor.fetchone()[0]
        columns_list = columns.split('@#')
        cursor.close()
        conn.close()
        return columns_list
    
    def _deal_llm_result(self, result):
        try:
            post_result, flag = extract_reasoning(result)
        except Exception:
            flag = False
            print('post_result:\n', post_result)
        while not isinstance(post_result, dict) or not flag:
            try:
                post_result = fix_conditions(result)
                flag = True
            except Exception:
                flag = False

            if not isinstance(post_result, dict) or not flag or 'not relevant' not in result:               
                prompt = '''
Output format error.Please directly output 'not relevant' or output the result in the following JSON format(no any explanation, no ```json```):
{
    "QueryTable-1":{   
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
                result = self.base_config.LLM.predict(self.base_config.args, prompt)
                # self.base_config.logger.log_info('-------------------------')
                # self.base_config.logger.log_info(f'user prompt:\n{prompt}')
                # self.base_config.logger.log_info(f'LLM response:\n{result}')
                if 'not relevant' in result:
                    return 'not relevant'
                post_result, flag = extract_reasoning(result)
        
        return post_result