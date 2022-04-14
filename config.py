import json

import torch
from utils import load_json

label2idx_path = "./data/label2idx.json"

train_data_path = "./data/train.txt"
dev_data_path = "./data/dev.txt"
test_data_path = "./data/test.txt"

config_path = "./bert_config/config.json"
model_path = "./bert_config/pytorch_model.bin"
vocab_path = "./bert_config/vocab.txt"

save_model_path = "./best_model.bin"

device = "cuda:3" if torch.cuda.is_available() else "cpu"
successor_layers = 3
hidden_size = 768
class_num = len(load_json(label2idx_path))
max_seq_len = 32
batch_size = 128
lr = 2e-5
epochs = 20



