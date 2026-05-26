from utils import ExperimentConfig, Tokenized_data
from trainer import train_vanilla, test_vanilla
from torch.utils.data import DataLoader

if __name__ == '__main__':
    exp_config = ExperimentConfig('analyze_vanilla5', '../configs/train_vanilla.yml', force_duplicate = True)

    dataset = Tokenized_data(exp_config.window_size, is_test=False)
    dataloader = DataLoader(dataset, batch_size=exp_config.batch_size)
    # load data
    train_vanilla(exp_config.device, 0, exp_config.learning_rate, dataloader, exp_config.vocab_size, exp_config.layer_num, exp_config.embed_dim, exp_config.heads_num, exp_config.hidden_dim, exp_config.window_size, exp_config.expert_num, exp_config.moe_at, exp_config.ckpt_dir, exp_config.max_steps, exp_config.save_steps, exp_config.warmup_steps)

    # test_dataloader = DataLoader(Tokenized_data(exp_config.window_size, is_test=True), batch_size=2)
    # test_vanilla(exp_config.device, 10000, test_dataloader, exp_config.vocab_size, exp_config.layer_num, exp_config.embed_dim, exp_config.heads_num, exp_config.hidden_dim, exp_config.window_size, exp_config.expert_num, exp_config.moe_at, exp_config.ckpt_dir)

