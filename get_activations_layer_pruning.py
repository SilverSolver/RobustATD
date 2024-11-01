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
batched_save = True
types = ['head_wise', 'layer_wise']

device = "cuda"

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    

if dataset_name == 'semeval':
    dataset = pd.read_csv("data/semEval/subtaskA_train_monolingual_cropped.csv")
    prompts, models, domains = utils.tokenized_semEval(dataset, tokenizer)

models_unique = ['davinci', 'bloomz', 'cohere', 'chatGPT', 'dolly', 'human']
domains_unique = ['wikipedia', 'wikihow', 'reddit', 'peerread', 'arxiv']


for l in range(12):
    print(f"Layer {l} pruned")
    model =  RobertaModel.from_pretrained('roberta-base', output_attentions=False).to(device)
    with torch.no_grad():
        model.encoder.layer[l].attention.output.dense.weight *= 0.0

    activations_dict = {m: {d: [] for d in domains_unique} for m in models_unique}
    
    for prompt, gen_model, gen_doman in tqdm(zip(prompts, models, domains), total = len(prompts)):
        activations = utils.get_roberta_last_hidden_state(model, prompt, device)
        activations_dict[gen_model][gen_doman].append(activations.mean(axis = 0))
    

    with open(f'embeddings/{model_name}_{dataset_name}_layer{l}_pruned.pkl', 'wb') as f:
        pickle.dump(activations_dict, f)
