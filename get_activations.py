import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
import pickle 
import utils


model_name = 'roberta'
dataset_name = 'semeval'


device = "cuda"

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model =  RobertaModel.from_pretrained('roberta-base', output_attentions=False).to(device)
    

if dataset_name == 'semeval':
    dataset = pd.read_csv("data/subtaskA_train_monolingual_cropped.csv")
    prompts, models, domains = utils.tokenized_semEval(dataset, tokenizer)

models_unique = ['davinci', 'bloomz', 'cohere', 'chatGPT', 'dolly', 'human']
domains_unique = ['wikipedia', 'wikihow', 'reddit', 'peerread', 'arxiv']


activations_dict = {m: {d: [] for d in domains_unique} for m in models_unique}

for prompt, gen_model, gen_doman in tqdm(zip(prompts, models, domains), total = len(prompts)):
    activations = utils.get_roberta_last_hidden_state(model, prompt, device)
    activations_dict[gen_model][gen_doman].append(activations.mean(axis = 0))


with open(f'embeddings/{model_name}_{dataset_name}.pkl', 'wb') as f:
    pickle.dump(activations_dict, f)

