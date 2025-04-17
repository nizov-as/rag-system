import configparser

class Config():
    """
        класс загрузки данных из config
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config_10.ini')
        #config.read('config_lc_bge.ini')
        #config.read('config_lc_bert.ini')
        self.temperature_qst_gen = float(config['temperature']['temperature_qst_gen'])
        self.temperature_fact_check = float(config['temperature']['temperature_fact_check'])
        self.temperature_response_gen = float(config['temperature']['temperature_response_gen'])

        self.prompt_template_qst = config['prompts']['prompt_template_qst']
        self.prompt_template_check = config['prompts']['prompt_template_check']
        self.prompt_template_response = config['prompts']['prompt_template_response']
        self.prompt_template_qst_response = config['prompts']['prompt_template_qst_response']
        self.system_prompt = config['prompts']['system_prompt']
        self.prompt_template_response_ext = config['prompts']['prompt_template_response_ext']
        
        self.base_path = config['path']['base_path']

        self.reranker_device = config['reranker']['reranker_device']
        self.doc_reranker_name = config['reranker']['doc_reranker_name']

        self.doc_emb_model_name = config['embedding']['doc_emb_model_name']
        self.emb_model_device = config['embedding']['emb_model_device']  

        #self.late_chunk_model_name = config['late_chunk_embedding']['late_chunk_model_name']
        #self.emb_model_device = config['late_chunk_embedding']['emb_model_device']
        
        self.max_seq_len = config['late_chunk_embedding']['max_seq_len']

    def reload(self):
        config = configparser.ConfigParser()
        config.read('config_10.ini')
        #config.read('config_lc_bge.ini')
        #config.read('config_lc_bert.ini')
        self.temperature_qst_gen = float(config['temperature']['temperature_qst_gen'])
        self.temperature_fact_check = float(config['temperature']['temperature_fact_check'])
        self.temperature_response_gen = float(config['temperature']['temperature_response_gen'])

        self.prompt_template_qst = config['prompts']['prompt_template_qst']
        self.prompt_template_check = config['prompts']['prompt_template_check']
        self.prompt_template_response = config['prompts']['prompt_template_response']
        self.prompt_template_qst_response = config['prompts']['prompt_template_qst_response']
        self.system_prompt = config['prompts']['system_prompt']
        self.prompt_template_response_ext = config['prompts']['prompt_template_response_ext']

        self.base_path = config['path']['base_path']

        self.reranker_device = config['reranker']['reranker_device']
        self.doc_reranker_name = config['reranker']['doc_reranker_name']

        self.doc_emb_model_name = config['embedding']['doc_emb_model_name']
        self.emb_model_device = config['embedding']['emb_model_device']    
    
        #self.late_chunk_model_name = config['late_chunk_embedding']['late_chunk_model_name']
        #self.emb_model_device = config['late_chunk_embedding']['emb_model_device']
        
        self.max_seq_len = config['late_chunk_embedding']['max_seq_len']
