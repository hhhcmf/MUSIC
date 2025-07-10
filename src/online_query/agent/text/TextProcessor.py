import re
class TextProcessor:
    def extract_entity_and_data_id(self, llm_result):
        """
        从llm_result格式中提取entity name和data id。
        
        参数:
            llm_result (str): 格式化的字符串，例如:
                entity name:<entity name> data id:<data id>
                entity name:<entity name> data id:<data id>
        
        返回:
            list: 包含(entity_name, data_id)的列表。
        """
        # 定义正则表达式匹配entity name和data id
        pattern = r'entity name:(.+?) data id:(.+)'
        matches = re.findall(pattern, llm_result, re.IGNORECASE)
        if not matches:      
            return []
        # 返回匹配结果的列表
        return [(entity_name.strip(), data_id.strip()) for entity_name, data_id in matches]