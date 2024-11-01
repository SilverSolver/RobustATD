from pathlib import Path

import numpy as np
import torch

from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel


import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from concept_erasure import LeaceEraser
import matplotlib as mpl
import pickle 
from sklearn.model_selection import train_test_split



task_names = ["bigram_shift",
              "coordination_inversion",
              "obj_number",
              "odd_man_out",
              "past_present",
              "sentence_length",
              "subj_number",
              "top_constituents",
              "tree_depth",
              "word_content"]


model_name = 'roberta'

if model_name == 'roberta':
    model_path = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model =  RobertaModel.from_pretrained(model_path, output_attentions=False).cuda()
elif model_name == 'phi2':
    MODEL_NAME = "microsoft/phi-2"
    model = AutoModel.from_pretrained(MODEL_NAME, 
                                                torch_dtype="auto", 
                                                trust_remote_code=True
                                                ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
elif model_name == 'bert':
    model_path = 'bert-base-uncased'
    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path, output_attentions=False).cuda()


def tokenize(text):
    return tokenizer(text, max_length=512, truncation = True, return_tensors='pt')

def loadFile(fpath):
    task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}
    
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split('\t')
            #task_data[tok2split[line[0]]]['X'].append(line[-1].split())
            task_data[tok2split[line[0]]]['X'].append(line[-1])
            task_data[tok2split[line[0]]]['y'].append(line[1])

    labels = sorted(np.unique(task_data['train']['y']))
    tok2label = dict(zip(labels, range(len(labels))))
    nclasses = len(tok2label)

    for split in task_data:
        for i, y in enumerate(task_data[split]['y']):
            task_data[split]['y'][i] = tok2label[y]
    return task_data



def get_embeddings(model, data):
    embs = []
    device = 'cuda'
    for i in tqdm(range(len(data))):
        text = data[i]
        inputs = tokenize(text)['input_ids'].to(device)
        with torch.no_grad():
            out = model(inputs)
        embeddings = out.last_hidden_state.detach().cpu().numpy().squeeze()
        embs.append(embeddings.mean(axis = 0))
    embs = np.vstack(embs)
    return embs

acc_dict = {}

for i, task_name in enumerate(task_names):

    print(f"{i}/{len(task_names)}: {task_name}")

    task_data = loadFile(f"data/probing/{task_name}.txt")

    texts = task_data['train']['X']
    Y = task_data['train']['y']

    #texts, Y, _, _ = train_test_split(texts, Y, stratify=Y, train_size=1000)

    X = get_embeddings(model, texts).astype(float)
    Y = task_data['train']['y']#[:1000]

    print(X.shape)

    lb  = LabelBinarizer().fit(Y)
    Y_t = lb.transform(Y)
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y_t)

    eraser = LeaceEraser.fit(X_t, Y_t)

    with open(f'erasers/{task_name}_{model_name}.pkl', 'wb') as f:
        pickle.dump(eraser, f)


    clf = LogisticRegression(max_iter=10000)
    clf.fit(X, Y)
    print(f'Accuracy before erasure: {clf.score(X,Y)}')

    X_ = eraser(X_t)
    clf.fit(X_.numpy(), Y)
    print(f'Accuracy after erasure: {clf.score(X_.numpy(),Y)}')

    print(f"Number of classes: {Y_t.shape[1]}")




