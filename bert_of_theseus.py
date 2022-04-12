# _*_ coding:utf-8 _*_


import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import transformers
from transformers import BertModel, BertConfig
from transformers import BertLayer
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from config import device
from utils import load_json
from crf import CRF
from crf2 import CRF as CRF2


class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=3):
        super().__init__()
        self.config = config
        self.prd_n_layer = config.num_hidden_layers
        self.scc_n_layer = scc_n_layer
        assert self.prd_n_layer % self.scc_n_layer == 0
        self.compress_ratio = self.prd_n_layer // self.scc_n_layer
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.scc_layer = nn.ModuleList([BertLayer(config) for _ in range(scc_n_layer)])
        self.gradient_checkpointing = False
        self.bernoulli = None

    def sample_bernoulli(self, rate=0.5):
        bernoulli = Bernoulli(rate)
        return bernoulli.sample()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.sample_bernoulli() == 1:  # 替换
                    inference_layers.append(self.scc_layer[i])
                else:
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i*self.compress_ratio + offset])
        else:
            inference_layers = self.scc_layer

        for i, layer_module in enumerate(inference_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Teacher(nn.Module):
    def __init__(self, config="./bert_config/config.json",
                 pretrained_path="./bert_config/pytorch_model.bin",
                 label2idx_path="./data/label2idx.json"):
        super(Teacher, self).__init__()
        self.label2idx = load_json(label2idx_path)
        self.num_cls = len(self.label2idx)
        self.bert_config = BertConfig.from_json_file(config)
        self.bert_model = BertModel(config=self.bert_config)
        self.bert_model.encoder = BertEncoder(config=self.bert_config)
        self.bert_model.from_pretrained(pretrained_path, config=self.bert_config)
        # self.bert_model.load_state_dict(torch.load(pretrained_path), strict=False)
        self.drop = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.crf = CRF(self.bert_config.hidden_size, self.num_cls)

    def __build_features(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(input_ids, attention_mask, token_type_ids)
        outputs = self.drop(outputs[0])
        return outputs

    def loss(self, input_ids, attention_mask, token_type_ids, tags):
        features = self.__build_features(input_ids, attention_mask, token_type_ids)
        loss = self.crf.loss(features, tags, masks=attention_mask)
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get the emission scores from the BiLSTM
        features = self.__build_features(input_ids, attention_mask, token_type_ids)
        scores, tag_seq = self.crf(features, attention_mask)
        return scores, tag_seq



class Teacher2(nn.Module):
    def __init__(self, config="./bert_config/config.json",
                 pretrained_path="./bert_config/pytorch_model.bin",
                 label2idx_path="./data/label2idx.json"):
        super(Teacher2, self).__init__()
        self.label2idx = load_json(label2idx_path)
        self.num_cls = len(self.label2idx)
        self.bert_config = BertConfig.from_json_file(config)
        self.bert_model = BertModel(config=self.bert_config)
        self.bert_model.encoder = BertEncoder(config=self.bert_config)
        self.bert_model.from_pretrained(pretrained_path, config=self.bert_config)
        # self.bert_model.load_state_dict(torch.load(pretrained_path), strict=False)
        self.drop = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.cls = nn.Linear(self.bert_config.hidden_size, self.num_cls)
        self.crf = CRF2(self.num_cls, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        sequence_outputs = outputs[0]
        sequence_outputs = self.drop(sequence_outputs)
        logits = self.cls(sequence_outputs)
        outputs = (logits, )
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss, ) + outputs
        return outputs





