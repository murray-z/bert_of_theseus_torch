# _*_ coding:utf-8 _*_


import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertModel
from torch.distributions.bernoulli import Bernoulli


# config = BertConfig.from_json_file("./bert_config/config.json")
# model = BertModel(config=config)
# print(model)
#
# embedding = model.embeddings(input_ids=torch.randint(1, 100, (2, 10)).type(torch.LongTensor),
#                              token_type_ids=torch.zeros((2, 10)).type(torch.LongTensor))
#
# print(embedding.shape)

for i in range(10):
    ratio = Bernoulli(torch.tensor([0.5]))
    print(ratio.sample()==1)


hidden_size = 768
class_num = 11

classfication_layer = nn.Linear(hidden_size, class_num)


class Predecessor(nn.Module):
    def __init__(self, pretrained_bert_name, classfication_layer):
        super(Predecessor, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_bert_name)
        self.predecessor = BertModel.from_pretrained(pretrained_bert_name)
        self.drop = self.config.hidden_dropout_prob
        self.classification_layer = classfication_layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.predecessor(input_ids, attention_mask, token_type_ids)
        last_hidden = self.drop(outputs[0])
        logits = self.classification_layer(last_hidden)
        return logits


class Successor(nn.Module):
    def __init__(self, config_path, num_hidden_layers, classification_layer):
        super(Successor, self).__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config["num_hidden_layers"] = num_hidden_layers
        self.successor = BertModel(config=self.config)
        self.classification_layer = classification_layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.predecessor(input_ids, attention_mask, token_type_ids)
        last_hidden = self.drop(outputs[0])
        logits = self.classification_layer(last_hidden)
        return logits


class Theseus(nn.Module):
    def __init__(self, predecessor_model, successor_model, classification_layer,
                 replace_rate=0.5):

        super(Theseus, self).__init__()

        self.classification_layer = classification_layer
        self.layers_per_module = predecessor_model.config.num_hidden_layers // \
                                 successor_model.config.num_hidden_layers
        self.predecessor_bert_model = predecessor_model.predecessor
        self.successor_bert_model = successor_model.successor

        # 冻结predecessor参数
        for child in self.predecessor_model.child():
            for param in child.parameters():
                param.requires_grad = False

        # 冻结classification_layer参数
        for param in self.classification_layer.parameters():
            param.requires_grad = False

    def replace_ratio(self):
        ratio = Bernoulli(torch.tensor([self.replacing_rate]))
        return ratio.sample()

    def forward(self, input_ids, attention_mask, token_type_ids):
        predecessor_hidden = self.predecessor_bert_model.embedding(input_ids, token_type_ids)
        successor_hidden = self.successor_bert_model.embedding(input_ids, token_type_ids)

        if self.training:
            pass
        else:
            pass


