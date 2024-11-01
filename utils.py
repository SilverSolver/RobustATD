import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def tokenized_semEval(dataset, tokenizer): 

    all_prompts = []
    all_models = dataset.model
    all_domains = dataset.source
    
    for i in tqdm(range(len(dataset))):
        prompt = dataset['text'][i]
        prompt = tokenizer(prompt, truncation = True, return_tensors = 'pt').input_ids
        all_prompts.append(prompt)

    return all_prompts, all_models, all_domains



def get_roberta_last_hidden_state(model, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        out = model(prompt)
        last_hidden_state = out.last_hidden_state.detach().cpu().numpy().squeeze()
    return last_hidden_state


def get_data(activations_dict, gen_model, gen_domain):
    model_activations = np.array(activations_dict[gen_model][gen_domain])
    human_activations = np.array(activations_dict['human'][gen_domain])
    X = np.vstack([human_activations, model_activations])
    Y = [0] * human_activations.shape[0] + [1] * model_activations.shape[0]
    return X, Y


def fit_and_score_output(activations_dict, subsets, class_weight, preprocessor = None):
    N = len(subsets)
    results = np.zeros((N, N))

    for i, (train_gen_model, train_gen_domain) in enumerate(subsets):
        X_train, Y_train = get_data(activations_dict, train_gen_model, train_gen_domain)

        #scaler = StandardScaler().fit(X_train)
        #X_train = scaler.transform(X_train)
        if preprocessor is not None:
            preprocessor = preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight).fit(X_train, Y_train)
        #param_grid = {'C': [0.01, 0.1, 1, 10]}
        #clf = GridSearchCV(LogisticRegression(max_iter=1000, class_weight=class_weight), param_grid=param_grid, cv = 4)
        #clf.fit(X_train, Y_train)

        for j, (test_gen_model, test_gen_domain) in enumerate(subsets):
            X_test, Y_test = get_data(activations_dict, test_gen_model, test_gen_domain)
            #X_test = scaler.transform(X_test)
            if preprocessor is not None:
                X_test = preprocessor.transform(X_test)
            pred = clf.predict(X_test)
            if class_weight == 'balanced':
                results[i,j] = balanced_accuracy_score(Y_test, pred)
            else:
                results[i,j] = accuracy_score(Y_test, pred)
            #X_train = train_activations#[:, -1, :]
            #X_test = test_activations#[:, -1, :]

    return results
