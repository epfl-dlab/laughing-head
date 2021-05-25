import tensorflow as tf
from transformers import BertForSequenceClassification

from transformers import (
    BertConfig,
    AutoTokenizer,
    BertTokenizer,
    AutoModelForSequenceClassification
)

import numpy as np
import pandas as pd


def convert_tuple_to_tensor(attention):
    attention_tensor = []
    for layer in attention:
        attention_tensor.append(layer[0].detach().numpy())
    return np.array(attention_tensor)


def get_bert_base_attention(df, pretrained, save_name, model=None):
    with tf.device('/device:cpu:0'):
        model_version = 'bert-base-uncased'
        # model_version = 'roberta-base'
        # model_version = 'distilbert-base-uncased'
        do_lower_case = True
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

        if pretrained == 'base':
            config = BertConfig.from_pretrained(model_version, num_labels=2,
                                                output_attentions=True, keep_multihead_output=True)
            # tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
            pytorch_model = BertForSequenceClassification.from_pretrained(model_version, config=config)
        elif pretrained == 'ss':
            pytorch_model = BertForSequenceClassification.from_pretrained(model, from_tf=True)
        else:
            pytorch_model = BertForSequenceClassification.from_pretrained(model)

        funny_attentions, serious_attentions = [], []
        for i in range(df.shape[0]):
            print("{} / {}".format(i, df.shape[0]))
            row = df.iloc[i]
            inputs = tokenizer.encode_plus(row['headline_original'],
                                           add_special_tokens=True, return_tensors='pt')

            outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
            _, funny_attention = outputs

            funny_attentions.append(convert_tuple_to_tensor(funny_attention))

            inputs = tokenizer.encode_plus(row['headline_unfunned'],
                                           add_special_tokens=True, return_tensors='pt')
            outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
            _, serious_attention = outputs

            serious_attentions.append(convert_tuple_to_tensor(serious_attention))

#     print(np.stack(funny_attentions).shape)
    np.savez(save_name + '_funny.npz', *funny_attentions)
    np.savez(save_name + '_serious.npz', *serious_attentions)


def get_bert_base_attention_auto(df, pretrained, save_name, model=None):
    with tf.device('/device:cpu:0'):
        # model_version = 'bert-base-uncased'
        model_version = 'roberta-base'
        # model_version = 'distilbert-base-uncased'
        do_lower_case = True
        tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

        if pretrained == 'base':
            config = BertConfig.from_pretrained(model_version, num_labels=2,
                                                output_attentions=True, keep_multihead_output=True)
            # tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
            pytorch_model = BertForSequenceClassification.from_pretrained(model_version, config=config)
        elif pretrained == 'ss':
            pytorch_model = AutoModelForSequenceClassification.from_pretrained(model, from_tf=True)
        else:
            pytorch_model = BertForSequenceClassification.from_pretrained(model)

        funny_attentions, serious_attentions = [], []
        for i in range(df.shape[0]):
            print("{} / {}".format(i, df.shape[0]))
            row = df.iloc[i]

            inputs = tokenizer.encode_plus(row['headline_original'],
                                           add_special_tokens=True, return_tensors='pt')

            if model_version.startswith('bert'):
                outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
            else:
                outputs = pytorch_model(inputs['input_ids'], output_attentions=True)

            _, funny_attention = outputs

            funny_attentions.append(convert_tuple_to_tensor(funny_attention))

            inputs = tokenizer.encode_plus(row['headline_unfunned'],
                                           add_special_tokens=True, return_tensors='pt')

            if model_version.startswith('bert'):
                outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
            else:
                outputs = pytorch_model(inputs['input_ids'], output_attentions=True)

            _, serious_attention = outputs

            serious_attentions.append(convert_tuple_to_tensor(serious_attention))

#     print(np.stack(funny_attentions).shape)
    np.savez(save_name + '_funny.npz', *funny_attentions)
    np.savez(save_name + '_serious.npz', *serious_attentions)


def get_tf_ss_attention_maps(df, pretrained_model, model_version, save_name):
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    pytorch_model = BertForSequenceClassification.from_pretrained(pretrained_model, output_attentions=True, from_tf=True)

    funny_attentions, serious_attentions = [], []
    for i in range(df.shape[0]):
        print("{} / {}".format(i, df.shape[0]))
        row = df.iloc[i]

        inputs = tokenizer.encode_plus(row['headline_original'],
                                       add_special_tokens=True, return_tensors='pt')
        outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
        _, funny_attention = outputs
        # print(funny_attention[0].detach().numpy().shape)
        # exit()

        funny_attentions.append(funny_attention[0].detach().numpy())

        inputs = tokenizer.encode_plus(row['headline_unfunned'],
                                       add_special_tokens=True, return_tensors='pt')
        outputs = pytorch_model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], output_attentions=True)
        _, serious_attention = outputs
        serious_attentions.append(serious_attention[0].detach().numpy())

    np.savez(save_name + '_funny.npz', *funny_attentions)
    np.savez(save_name + '_serious.npz', *serious_attentions)


import random


def replace(fun_sent, ser_sent, voc):
    to_replace = []
    for i, tok in enumerate(fun_sent):
        if tok not in ser_sent:
            to_replace.append(i)

    new_sent = fun_sent.copy()
    for i in to_replace:
        new_sent[i] = random.choice(voc)
    return ' '.join(new_sent)


def random_perturbation_df(df):
    funny_headlines = df['headline_original']
    serious_headlines = df['headline_unfunned']

    vocabulary = []
    for fun, ser in zip(funny_headlines, serious_headlines):
        vocabulary.extend(fun.split(' '))
        vocabulary.extend(ser.split(' '))

    vocabulary = list(set(vocabulary))

    modified_funny_headlines = []
    for fun, ser in zip(funny_headlines, serious_headlines):
        mod_fun = replace(fun.split(' '), ser.split(' '), vocabulary)
        modified_funny_headlines.append(mod_fun)

    df['headline_original'] = modified_funny_headlines

    return df


if __name__ == '__main__':
    test_df = pd.read_csv('Analysis/test_set_paired.csv')
    # test_df = random_perturbation_df(test_df)

    # get_bert_base_attention(test_df, pretrained='base', save_name='Analysis/att_map_BERT_base_full')
    # get_bert_base_attention_auto(test_df, pretrained='ss', save_name='Analysis/RoBERTa-base-ss-00-attnmaps/att_map_BERT_ss_full', model='Models/RoBERTa-ss/')
    get_bert_base_attention(test_df, pretrained='ss', save_name='Analysis/BERT-base-ss-origins-attnmaps/att_map_BERT_ss_full',
                            # model='Models/BERT-base-ss-00/')
                            model='/Users/peyrardm/Documents/LaughingHeads/BERT-models/BERT_ss')
    # get_bert_base_attention(test_df, pretrained='siamese', save_name='Analysis/att_map_BERT_siamese_full', model='BERT-models/BERT_siamese/')

    # get_tf_ss_attention_maps(test_df, pretrained_model='Models/BERT-base-ss-00',
    #                                   model_version='bert-base-uncased',
    #                                   save_name='Analysis/BERT-base-ss-00-attnmaps/att_map_BERT_ss_full')

    # get_tf_ss_attention_maps(test_df, pretrained_model='Models/BERT-base-ss-01',
    #                          model_version='bert-base-uncased',
    #                          save_name='Analysis/BERT-base-ss-01-attnmaps/att_map_BERT_ss_full')
