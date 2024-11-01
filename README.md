# Robust AI-Generated Text Detection by Restricted Embeddings

Welcome to the official repository for the paper [**Robust AI-Generated Text Detection by Restricted Embeddings**](https://arxiv.org/abs/2410.08113). The essential code for replicating experiments is provided here. 

_Growing amount and quality of AI-generated texts makes detecting such content more difficult. In most real-world scenarios, the domain (style and topic) of generated data and the generator model are not known in advance. In this work, we focus on the robustness of classifier-based detectors of AI-generated text, namely their ability to transfer to unseen generators or semantic domains. We investigate the geometry of the embedding space of Transformer-based text encoders and show that clearing out harmful linear subspaces helps to train a robust classifier, ignoring domain-specific spurious features. We investigate several subspace decomposition and feature selection strategies and achieve significant improvements over state of the art methods in cross-domain and cross-generator transfer. Our best approaches for head-wise and coordinate-based subspace removal increase the mean out-of-distribution (OOD) classification score by up to 9% in particular setups for RoBERTa and BERT embeddings._

![intuition](https://github.com/SilverSolver/RobustATD/blob/main/emnlp_intuition.png?raw=true)

**Python files**:
- `utils.py` : some useful functions
- `evaluation.py` : contains an evaluator class for SemEval
- `get_activations.py` : getting the activations (last hidden state) for SemEval
- `get_activations_layer_pruning.py` : getting the activations (last hidden state) for SemEval when layer is pruned
- `get_activations_selected_pruning.py` : getting the activations (last hidden state) for SemEval when selected heads are pruned
- `fit_eraser_probing_tasks.py` : fitting concept eraser on probing tasks

**Notebooks**:
- `example_evaluation.ipynb` : an example of calculating scores for SemEval
- `finding_bad_components_RoBERTa_code.ipynb` : selecting the coordinates with greedy search (updated comparing to the paper)
- `heads_prunning.ipynb` : selecting heads with greedy search on GPT3D dataset
- `layer_prunning.ipynb` : an example of pruning an entire layer of attention heads
- `PhDim_ATD.ipynb` : examining PHDim method

**Folders**:
- `cache` : folder to store temporary files with models embeddings
- `data` : contains cropped SemEval, probing tasks and GPT3D dataset
- `embeddings` : folder for storing embeddings 
- `erasers` : folder for storing erasers

---

# Erratum

After the acceptance of the camera-ready version of our paper to **Findings of EMNLP 2024**, we found a mistake in our code for **Selected coordinates** method. Namely, in the experiments with this particular method validation subsets of "Wikipedia" and "Reddit" subdomains of GPT-3D were intersecting with test subsets on the same subdomains, which is obviously incorrect; fortunately, no such problem for other methods was found. We fixed this mistake and received new results for Selected coordinates method. This is the list of all updated results:

Updated results for Figure 3 e in paper (accuracy):

![figure_3_e](https://github.com/SilverSolver/RobustATD/blob/main/best_set_of_coords.png?raw=true)

Updated results for Table 3 in paper (accuracy):

| RoBERTa | CD | CM | CD | CM | CA |
| --- | --- | --- | --- | --- | --- |
| Baseline | 73.0 | 82.8|  84.1| 71.0| 70.1 |
| ~~Selected coordinates~~ | ~~74.5~~ | ~~82.6~~ | ~~85.4~~ | ~~71.9~~ | ~~72.8~~ |
| Selected coordinates | **77.1** | **84.2** | **88.2** | 72.2 | 69.7 |

Updated results for Table 4 in paper (accuracy):

| | BERT | | | Phi-2 | | |
| --- | --- | --- | --- | --- | --- | --- |
|   | CD | CM | CA | CD | CM | CA |
| Baseline | 82.4 | 81.9 | 71.1 | 92.2 | 92.3 | 86.7 | 92.8 | 88.5 | 80.5 |
| ~~Selected coordinates~~ | ~~92.1~~ | ~~88.0~~ | ~~85.2~~ | ~~93.1~~ | ~~89.9~~ | ~~86.7~~ |
| Selected coordinates | **88.0** | **88.9** | **80.0** | 91.3 | 90.5 | 80.9 |

Here, we repeated the values for the baseline from paper for comparison and highlighted those values, that are better than other methods, with **bold**. Old, incorrect results are written in ~~strikethrough text~~.

Note that the **general conclusions of the paper still remain valid** and don't change with these results; however, particular values change quite significantly, so if you want to compare your own method of Artificial Text Detection with Selected coordinates method, it will be more correct and reasonable to use these updated values for comparison. Also note that the results of Selected Coordinates and Selected Head methods are in general dependent of the choice of the validation set, and you need to take it into account when trying to reproduce the results.

We are sorry for possible confusion.

Cite us as:

```
@misc{kuznetsov2024robustaigeneratedtextdetection,
      title={Robust AI-Generated Text Detection by Restricted Embeddings}, 
      author={Kristian Kuznetsov and Eduard Tulchinskii and Laida Kushnareva and German Magai and Serguei Barannikov and Sergey Nikolenko and Irina Piontkovskaya},
      year={2024},
      eprint={2410.08113},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.08113}, 
}
```
(ACL citation will be available after adding our paper to ACL Anthology).
