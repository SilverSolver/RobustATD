import utils
from itertools import product
import numpy as np
import seaborn as sns
import pandas as pd


class Evaluator():

    def __init__(self, concat_axis = None):
        #self.act_type = act_type  #output, layers, heads
        self.concat_axis = concat_axis

    def prepare_data(self, activations_dict):
        human_activations = activations_dict['human']
        model_activations = {m: activations_dict[m] for m in activations_dict.keys() if m != 'human'}
        gen_models = model_activations.keys()
        gen_domains = human_activations.keys()

        if self.concat_axis is None:
            subsets = list(product(gen_models, gen_domains))
            self.axis_labels = [m+'_'+d for (m, d) in subsets]
        elif self.concat_axis == 'model':
            activations_dict = {'u_model': {d: np.vstack([model_activations[m][d] for m in gen_models]) for d in gen_domains}}
            activations_dict['human'] = human_activations
            subsets = list(product(['u_model'], gen_domains))
            self.axis_labels = [d for (m, d) in subsets]
        elif self.concat_axis == 'domain':
            activations_dict = {m: {'u_domain': np.vstack([model_activations[m][d] for d in gen_domains])} for m in gen_models}
            activations_dict['human'] = {'u_domain': np.vstack([human_activations[d] for d in gen_domains])}
            subsets = list(product(gen_models, ['u_domain']))
            self.axis_labels = [m for (m, d) in subsets]
        return activations_dict, subsets
    

    def fit(self, activations_dict, class_weight = None, preprocessor = None):
        activations_dict, subsets = self.prepare_data(activations_dict)
        self.scores = utils.fit_and_score_output(activations_dict, subsets, class_weight, preprocessor) * 100

    def plot_heatmap(self, ax = None, **kwargs):
        scores_df = pd.DataFrame(self.scores, index=self.axis_labels, columns=self.axis_labels).round(1)
        sns.heatmap(scores_df, square=True, annot=True, ax = ax, fmt='g', **kwargs)

    def get_activations_dict(self, activations_dict):
        activations_dict, subsets = self.prepare_data(activations_dict)
        return activations_dict, subsets

    def get_scores(self):
        return self.scores
    
    def set_scores(self, scores):
        self.scores = scores
