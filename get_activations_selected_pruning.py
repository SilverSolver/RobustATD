import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import utils
from transformers import AutoTokenizer
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import pickle 

model_name = 'roberta'
dataset_name = 'semeval'


device = "cuda"

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    

if dataset_name == 'semeval':
    dataset = pd.read_csv("data/semEval/subtaskA_train_monolingual_cropped.csv")
    prompts, models, domains = utils.tokenized_semEval(dataset, tokenizer)

models_unique = ['davinci', 'bloomz', 'cohere', 'chatGPT', 'dolly', 'human']
domains_unique = ['wikipedia', 'wikihow', 'reddit', 'peerread', 'arxiv']

heads_to_remove = {0: [7, 4, 9, 11, 8], 1: [6, 11, 0, 10], 2: [0, 3, 8], 3: [0, 1, 6, 11], 4: [3, 4, 5, 0, 11], 5: [1, 3, 7], 6: [6, 4], 8: [3], 9: [8, 2, 6, 9, 10, 5], 10: [1], 7: [6, 11], 11: [4]}

model =  RobertaModel.from_pretrained('roberta-base', output_attentions=False).to(device)
model._prune_heads(heads_to_remove)



activations_dict = {m: {d: [] for d in domains_unique} for m in models_unique}

for prompt, gen_model, gen_doman in tqdm(zip(prompts, models, domains), total = len(prompts)):
    activations = utils.get_roberta_last_hidden_state(model, prompt, device)
    activations_dict[gen_model][gen_doman].append(activations.mean(axis = 0))

with open(f'embeddings/{model_name}_{dataset_name}_selected_pruning.pkl', 'wb') as f:
    pickle.dump(activations_dict, f)
