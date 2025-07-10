from data.MMQA.preprocess import get_tableCol_by_col
from online_query.agent.base import baseAgent
from online_query.agent.table.prompt import TablePrompt
from online_query.agent.table.TableProcessor import TableProcessor, SQLiteConnectionPool
import sqlite3
import time
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

class TableParser(baseAgent):
    def __init__(self, base_config, kg_config):   
        super().__init__(base_config)
        self.kg_config = kg_config
        self.table_processor = TableProcessor(base_config)
        self.prompt_generator = TablePrompt()
        self.conn_pool = SQLiteConnectionPool(self.table_processor.db_path, pool_size=10)  # 设置连接池大小
        self.query_cache = {} 

    async def inference(self, question, userful_information, table_entity_data_list, judge_object_question = ''):
        '''
        针对表格模态的推理Query-Table Inference
        输出:table_name表名str, select_columns(以@#为分隔符的字符串), conditions:dict(value:column), output:str
        '''
        start_time = time.time()
        table_records_data_list = []
        self.base_config.logger.log_info("----------agent:TableParser被激活--------\n")
        
        
        table_flag = 0
        for i in range(0, 3):
            
            reasoning_prompt = self.prompt_generator.create_pipeline_reason_prompt(question, userful_information, table_entity_data_list)
            self.base_config.table_llm.init_message()
            table_judge_start_time = time.time()
            result, flag = await self.base_config.table_llm.predict_async(self.base_config.args, reasoning_prompt)
            if not flag:
                raise ValueError('openai content error')
            table_judge_time = time.time()-table_judge_start_time
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info("step 1:调用QueryTable获取表格数据\n")
                self.base_config.logger.log_info(f"User prompt:{reasoning_prompt}\n")
                self.base_config.logger.log_info(f"LLM response:{result}\n")
            if 'not relevant' not in result.lower():
                post_data = self.table_processor._deal_llm_result(result)
                if post_data != 'not relevant':
                    table_flag = 1
                    break
        if table_flag == 0:
            inference_time =  time.time()-start_time
            # print(f'table all time:{time.time()-start_time}')
            return [], [], inference_time
        
        
        post_data = self.table_processor._deal_llm_result(result)

        tasks = []
        for key, input_data in post_data.items():
            task = asyncio.create_task(self.process_input_data(input_data, table_entity_data_list))
            tasks.append(task)
        table_qa_start_time = time.time()
        results = await asyncio.gather(*tasks)
        table_qa_time = time.time()-table_qa_start_time
        for result in results:
            table_records_data_list.extend(result)

        inference_time = table_judge_time + table_qa_time
        # print(f'table all time:{time.time()-start_time}')

        return table_records_data_list, table_entity_data_list, inference_time

    async def process_input_data(self, input_data, table_entity_data_list):
        table_name, table_desp, select_columns, conditions, output, rows, flag = self.deal_reason(input_data, table_entity_data_list)
        table_records_data_list = []
        data_id = table_entity_data_list[0].data_id
        if self.base_config.args.SHOW_LOGGER:
            self.base_config.logger.log_info(f"Table data:\ndata id:{data_id}\ntable name:{table_name}\nconditions:{conditions}\nselect columns:{select_columns}\ntable records:\n{output}\n")
        conditions_json = json.dumps(conditions, sort_keys=True) 
        table_records_data_list.append((table_name.upper(), table_desp.upper(), select_columns.upper(), conditions_json.upper(), output.upper(), tuple(tuple(row) for row in rows)))
        return table_records_data_list
   

    def deal_reason(self, input_data, table_entity_data_list):
        #处理llm给出的api的输入，解析表名、查询列名、以及条件，确保输入的条件以及查询的列名是正确的
        select_column_list = []
        input_conditions = {}
        if input_data:
            table_name = table_entity_data_list[0].table_name
            title = table_entity_data_list[0].title
            
            if title:
                name = table_name[len(title)+1:]
                table_desp = f'This table records {name} of {title}'
            else:
                table_desp = ''

            select_column = input_data.get('select columns', '')
            if not isinstance(select_column, str):
                select_column = table_entity_data_list[0].columns
            select_column_list = select_column.split('@#')
            select_columns = ', '.join(select_column_list)
            input_conditions = input_data.get('conditions',{})
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f'table api args:{table_name}\n{select_column}\n{input_conditions}\n')

        table_flag = False
        
        conditions = {}
        table_name = table_entity_data_list[0].table_name
        table_columns = table_entity_data_list[0].columns
        table_column_list = table_entity_data_list[0].columns.split('@#')
        if len(select_column_list) <= 1 or not set(select_column_list).issubset(set(table_column_list)):
            select_columns = ', '.join(table_column_list)
            select_column = table_columns
                    
        if table_entity_data_list:
            for table_entity_data in table_entity_data_list:
                is_column_value = table_entity_data.is_column_value
            
                if is_column_value:
                    belong_column = table_entity_data.column
                    column_value = table_entity_data.entity_name
                    if input_conditions:#确保llm给定的过滤条件是确定正确的，而不是胡编乱造的
                        if column_value in input_conditions.keys() and input_conditions[column_value] == belong_column:
                            conditions[column_value] = belong_column
                        if belong_column in input_conditions.keys() and input_conditions[belong_column] == column_value:
                            conditions[column_value] = belong_column
                else:
                    if input_conditions:
                        for key, value in input_conditions.items():
                            if key in table_column_list:
                                conditions[value] = key
                            if value in table_column_list:
                                conditions[key] = value
                                
        # conn = sqlite3.connect(self.table_processor.db_path)
        # cursor = conn.cursor()
        # sql_query = f'SELECT {select_columns} FROM {table_name}'
        select_columns = ', '.join(table_column_list)
        select_column = table_columns
        api_output, rows, flag = self.QueryTable(table_name, select_columns, conditions)
        

        return table_name, table_desp, select_column, conditions, api_output, rows, flag
            
          

    def QueryTable(self, table_name, select_columns, conditions):
        query_key = (table_name, tuple(select_columns), tuple(conditions.items()))
        if query_key in self.query_cache:
            result, rows = self.query_cache[query_key]
            return result, rows, True
        
        conn = self.conn_pool.get_connection()
        cursor = conn.cursor()

        result = ''
        if self.base_config.args.dataset == 'manymodalqa':
            sql_query = f'SELECT {select_columns} FROM {table_name}'
            # print(sql_query)
            table_id = int(table_name)
            cols_list = self.table_processor.get_cols_by_table_id(table_id)
            sql_element_list = sql_query.split(' ')
            select_columns = list(set(cols_list).intersection(sql_element_list))
            
            where_query =''
            if conditions:
                where_query += ' WHERE '
                condition_list = [f"{column_name} = '{column_value}'" for column_value, column_name in conditions.items()]
                where_query += ' OR '.join(condition_list)
            sql_query = f'SELECT {select_columns} FROM {table_name}{where_query}'

            cursor.execute(sql_query)
            rows = cursor.fetchall()
            if not rows:
                sql_query = f'SELECT {select_columns} FROM {table_name}'
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                # result += 'There are no records matching specific criteria, so you can retrieve all records from the table without any conditions'
                # return result, False
            
            column_names = [description[0] for description in cursor.description]
            result += f"column name: {'|'.join(column_names)}\n"
            for idx, row in enumerate(rows, start=1):
                result += f"row{idx}: {'|'.join(map(str, row))}\n"
            self.query_cache[query_key] = (result, rows)
            self.conn_pool.release_connection(conn)
            return result, rows, True

        elif self.base_config.args.dataset == 'mmqa':
        # result = f'API:QueryTable Input:{sql_query} Output:\n'
            result = ''
            sql_query = f'SELECT {select_columns} FROM {table_name}'
            # print(sql_query)
            
            sql_table_name = table_name
            table_id = self.table_processor.get_tableId_by_name(sql_table_name)
            if not table_id:
                result += f'The database does not have this table "{sql_table_name}".Please carefully read the given table information and then query the correct table.'
                return result, '', False
            
            cols_list = self.table_processor.get_cols_by_table_id(table_id)
            sql_element_list = sql_query.split(' ')
            sql_cols = list(set(cols_list).intersection(sql_element_list))

            col_to_table, table_to_col = {}, {}
            for col in cols_list:
                table_col = get_tableCol_by_col(col, table_id)
                col_to_table[col] = table_col
                table_to_col[table_col] = col
            
            after_table_name = table_name.replace(sql_table_name, f'table_{table_id}')

            cols_list = sorted(cols_list, key=len, reverse=True)
            for sql_col in cols_list:
                select_columns = select_columns.replace(sql_col, col_to_table[sql_col])
            
            # print(col_to_table)
            where_query =''
            if conditions:
                # print(conditions)
                where_query += ' WHERE '
                condition_list = [f"{col_to_table[column_name.replace(' ', '_')]} = '{column_value}'" for column_value, column_name in conditions.items()]
                where_query += ' OR '.join(condition_list)
            sql_query = f'SELECT {select_columns} FROM {after_table_name}{where_query}'
            # print('after sql:', sql_query)
            if self.base_config.args.SHOW_LOGGER:
                self.base_config.logger.log_info(f"QueryTable SQL:{sql_query}\n")
            try:
                cursor.execute(sql_query)
                # rows = cursor.fetchall()
                # if not rows:
                #     result += 'There are no records matching specific criteria, you can retrieve all records from the table without any conditions'
                #     return result, False

                # column_names = [table_to_col[description[0]] for description in cursor.description]
                # # result += f"column name: {'|'.join(column_names)}\n"
                # for idx, row in enumerate(rows, start=1):
                #     result += f"row{idx}: {'|'.join(map(str, row))}\n"
            
            except Exception as e:
                # print(e)
                sql_query = f'SELECT {select_columns} FROM {after_table_name}'
                if self.base_config.args.SHOW_LOGGER:
                    self.base_config.logger.log_info(f"QueryTable SQL:{sql_query}\n")
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            if not rows:
                # result += 'There are no records matching specific criteria, you can retrieve all records from the table without any conditions'
                # self.conn_pool.release_connection(conn)
                sql_query = f'SELECT {select_columns} FROM {after_table_name}'
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                # return result, '', False

            column_names = [table_to_col[description[0]] for description in cursor.description]
            # result += f"column name: {'|'.join(column_names)}\n"
            for idx, row in enumerate(rows, start=1):
                result += f"row{idx}: {'|'.join(map(str, row))}\n"
            
            self.query_cache[query_key] = (result, rows)
            self.conn_pool.release_connection(conn)
            return result, rows, True
        else:
            raise f"Table parser doesn't support {self.base_config.args.dataset}"