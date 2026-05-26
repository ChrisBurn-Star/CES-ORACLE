import yaml
import os
import shutil

class ExperimentConfig:
    def __init__(self, task_name, exp_config_path, force_duplicate = False):
        self.task_name = task_name

        self.ckpt_dir = f'../checkpoints/{task_name}'
        if os.path.exists(self.ckpt_dir):
            if force_duplicate:
                print(f'Warning: Task {task_name} already exists, will overwrite it.')
            else:
                raise ValueError(f'Error: Task {task_name} already exists.')
        else:
            os.makedirs(self.ckpt_dir)

        # copy config file to ckptdir
        shutil.copy(exp_config_path, f'{self.ckpt_dir}/{task_name}.yml')
        with open(exp_config_path, 'r') as stream:
            exp_config = yaml.safe_load(stream)

        self.model = exp_config['model']
        self.device = exp_config['trainer config']['device']
        self.batch_size = exp_config['trainer config']['batch_size']
        self.max_steps = exp_config['trainer config']['max_steps']
        self.save_steps = exp_config['trainer config']['save_steps']
        self.log_steps = exp_config['trainer config']['log_steps']
        self.learning_rate = exp_config['trainer config']['learning_rate']
        self.warmup_steps = exp_config['trainer config']['warmup_steps']

        shutil.copy(f'../configs/{self.model}.yml', f'{self.ckpt_dir}/{self.model}.yml')
        with open(f'../configs/{self.model}.yml', 'r') as stream:
            model_config = yaml.safe_load(stream)
        self.expert_num = model_config['MoE config']['expert_num']
        self.hidden_dim = model_config['MoE config']['hidden_dim']
        self.moe_at = model_config['MoE config']['moe_at']

        shutil.copy(f'../configs/{model_config["GPT config"]}.yml', f'{self.ckpt_dir}/{model_config["GPT config"]}.yml')
        with open(f'../configs/{model_config['GPT config']}.yml', 'r') as stream:
            GPT_config = yaml.safe_load(stream)
        self.vocab_size = GPT_config['vocab_size']
        self.embed_dim = GPT_config['embed_dim']
        self.hidden_dim = GPT_config['hidden_dim']
        self.layer_num = GPT_config['layer_num']
        self.heads_num = GPT_config['heads_num']
        self.window_size = GPT_config['window_size']

    
    # def __getattribute__(self, name: str):
    #     config_dict = super().__getattribute__('config')

    #     if name in config_dict:
    #         return config_dict[name]
    #     else:
    #         return super().__getattribute__(name)