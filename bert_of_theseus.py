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



class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=6):
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
        self.tag_to_ix = load_json(label2idx_path)
        num_cls = len(self.tag_to_ix)
        self.bert_config = BertConfig.from_json_file(config)
        self.bert_model = BertModel(config=self.bert_config)
        self.bert_model.encoder = BertEncoder(config=self.bert_config)
        self.bert_model.from_pretrained(pretrained_path, config=self.bert_config)
        # self.bert_model.load_state_dict(torch.load(pretrained_path), strict=False)
        self.drop = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.cls = nn.Linear(self.bert_config.hidden_size, num_cls)
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.tagset_size = len(self.tag_to_ix)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000

        self.transitions.to(device)


    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000., device=device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp

                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        batch_size = tags.size()[0]
        score = torch.zeros([1]*batch_size, device=device).unsqueeze(1)

        start_tags = torch.tensor([self.tag_to_ix[self.START_TAG]] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        tags = torch.cat([start_tags, tags], dim=1)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _get_bert_features(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(input_ids, attention_mask, token_type_ids)
        hiddens = self.drop(outputs[0])
        logits = self.cls(hiddens)
        return logits


    def neg_log_likelihood(self, input_ids, attention_mask, token_type_ids, tags):
        feats = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        score, tag_seq = self._viterbi_decode(logits)
        return score, tag_seq



class Teacher2(nn.Module):
    def __init__(self, num_cls=7):
        super(Teacher2, self).__init__()
        self.bert_config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.drop = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.cls = nn.Linear(self.bert_config.hidden_size, num_cls)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hiddens = self.drop(outputs[0])
        logits = self.cls(hiddens)
        return logits





