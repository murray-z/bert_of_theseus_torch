# _*_ coding:utf-8 _*_

import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel
from torch.distributions.bernoulli import Bernoulli
from config import *

classification_layer = nn.Linear(hidden_size, class_num)


class Predecessor(nn.Module):
    def __init__(self, config_path, pretrained_model_path, classification_layer):
        super(Predecessor, self).__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.predecessor = BertModel.from_pretrained(pretrained_model_path,
                                                     config=self.config)
        self.classification_layer = classification_layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.predecessor(input_ids, attention_mask, token_type_ids)
        logits = self.classification_layer(outputs[0])
        return logits


class Successor(nn.Module):
    def __init__(self, config_path, classification_layer):
        super(Successor, self).__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.successor = BertModel(config=self.config)
        self.classification_layer = classification_layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.successor(input_ids, attention_mask, token_type_ids)
        logits = self.classification_layer(outputs[0])
        return logits


class Theseus(nn.Module):
    def __init__(self, predecessor_model, successor_model, classification_layer,
                 replace_rate=0.5):

        super(Theseus, self).__init__()

        self.replace_rate = replace_rate
        self.classification_layer = classification_layer
        self.predecessor_num_hidden_layers = predecessor_model.config.num_hidden_layers
        self.successor_num_hidden_layers = successor_model.config.num_hidden_layers
        self.layers_per_module = self.predecessor_num_hidden_layers // self.successor_num_hidden_layers
        self.predecessor = predecessor_model
        self.successor = successor_model
        self.predecessor_bert_model = self.predecessor.predecessor
        self.successor_bert_model = self.successor.successor

        # 冻结predecessor参数
        for param in predecessor_model.parameters():
            param.requires_grad = False

        # 冻结classification_layer参数
        for param in self.classification_layer.parameters():
            param.requires_grad = False

    def random_choice(self, a, b):
        ratio = Bernoulli(torch.tensor([self.replace_rate])).sample()
        return ratio * a + (1 - ratio) * b

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_shape = input_ids.size()
        device = input_ids.device
        attention_mask = self.predecessor_bert_model.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 device=device)
        predecessor_embedding_hidden = self.predecessor_bert_model.embeddings(input_ids, token_type_ids)
        successor_embedding_hidden = self.successor_bert_model.embeddings(input_ids, token_type_ids)

        # 随机选择embedding输出
        hidden = self.random_choice(predecessor_embedding_hidden, successor_embedding_hidden)

        if self.training:
            for i in range(self.successor_num_hidden_layers):
                predecessor_output = hidden
                for j in range(self.layers_per_module):
                    predecessor_output = \
                    self.predecessor_bert_model.encoder.layer[i * self.layers_per_module + j]\
                        (hidden_states=predecessor_output, attention_mask=attention_mask)[0]
                successor_output = self.successor_bert_model.encoder.layer[i](hidden_states=hidden,
                                                                              attention_mask=attention_mask)[0]
                hidden = self.random_choice(predecessor_output, successor_output)
            logits = self.classification_layer(hidden)
            return logits
        else:
            outputs = self.successor_bert_model(input_ids, attention_mask, token_type_ids)
            logits = self.classification_layer(outputs[0])
            return logits
