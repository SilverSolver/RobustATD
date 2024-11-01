The essential code for replicating experiments is provided here. Here are some explanations:

- utils.py : some useful functions
- evaluation.py : contains an evaluator class for SemEval

- get_activations.py : getting the activations (last hidden state) for SemEval
- get_activations_layer_pruning.py : getting the activations (last hidden state) for SemEval when layer is pruned
- get_activations_selected_pruning.py : getting the activations (last hidden state) for SemEval when selected heads are pruned
- fit_eraser_probing_tasks.py : fitting concept eraser on probing tasks

Notebooks:
- example_evaluation.ipynb : an example of calculating scores for SemEval
- finding_bad_components_RoBERTa_code.ipynb : selecting the coordinates with greedy search (updated comparing to the paper)
- heads_prunning.ipynb : selecting heads with greedy search on GPT3D dataset
- layer_prunning.ipynb : an example of pruning an entire layer of attention heads
- PhDim_ATD.ipynb : examining PHDim method


Folders:
- cache : folder to store temporary files with models embeddings
- data : contains cropped SemEval, probing tasks and GPT3D dataset
- embeddings : folder for storing embeddings 
- erasers : folder for storing erasers
