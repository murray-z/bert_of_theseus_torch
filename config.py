import torch


label2idx_path = "./data/label2idx.json"
train_data_path = "./data/train.txt"
dev_data_path = "./data/dev.txt"
test_data_path = "./data/test.txt"
predecessor_config_path = "./bert_config/config.json"
predecessor_model_path = "./bert_config/pytorch_model.bin"
successor_config_path = "./bert_config/successor_config.json"
vocab_path = "./bert_config/vocab.txt"
best_predecessor_model_path = "./model/best_predecessor.pth"
best_successor_model_path = "./model/best_successor.pth"
best_theseus_model_path = "./model/best_theseus.pth"
device = "cuda:3" if torch.cuda.is_available() else "cpu"
successor_layers = 3
hidden_size = 768
class_num = 7
max_seq_len = 64
batch_size = 128
lr = 5e-5
epochs = 10



