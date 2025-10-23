import yaml
import os
from PIL import Image

CONFIG_FILE = '/home/cmf/multiQA/config/config.yaml'

class ConfigLoader:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_image_path(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['image_path']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    def get_final_image(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['final_image_path']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    def get_text_path(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['text_path']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")
        
    def get_table_path(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['table_path']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    def get_dev_path(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['dev_path']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    def get_database_config(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['database']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")
    
    def get_kg_config(self, dataset_name):
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]['kg']
        else:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")
    def get_user_settings(self):
        return self.config.get('user_settings', {})

config_file = '/home/cmf/multiQA/config/config.yaml'
config_loader = ConfigLoader(config_file)
IMAGE_PATH = config_loader.get_final_image('MMQA')
MMQA_DB = config_loader.get_database_config('MMQA')