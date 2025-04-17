import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_parse import Config

class LLM():
    """
        класс LLM
    """
    def __init__(self):
        self.mod_name = 'Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.mod_name,
            torch_dtype="auto",
            device_map="auto",
            output_hidden_states=False,
            do_sample=True,
            temperature=0.2  
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.mod_name)


    def invoke(self, prmt):
        search_config = Config()
        sys_prompt = search_config.system_prompt
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prmt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                truncation=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            full_generated_ids = self.llm.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            new_generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, full_generated_ids)]
            res0 = self.tokenizer.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
        except Exception as ex:
            print(ex)
            res0 = None
        return res0

    def change_temp(self, temp):
        # self.llm.model.config.temperature = temp
        self.llm.generation_config.temperature = temp
