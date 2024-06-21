# meta-tail2vec+

This repository is the official implementation of our paper [Locality-Aware Tail Node Embeddings on Homogeneous and Heterogeneous Networks](https://zemin-liu.github.io/papers/Locality-aware-tail-node-embeddings-on-homogeneous-and-heterogene.pdf), which is published in TKDE 2023.

## Repository Structure (in each dataset folder)


- /dataset/:
    - sample_metagraph_stats: Input file containing metagraph statistics. Here we only include a small sample of the dblp graph, which contains metagraphs up to size 4 only to limit the size of the file. Each row represents a relationship between a metagraph and two nodes. The first two columns are nodes' id, the third column is metagraph's id (starting with m) and the last column is the frequency of the metagraph instances appearing with the two nodes (staring with f).
- main/:
- main.py: Implementation of model.
- maml_8.py: Implementation of maml.
- data_generator_f.py: Preprocess the dataset for maml.
- prep_dblp_classification/prep_dblp_prediction.py: Preprocess the raw dataset.
- task/:
	- atask_multiclass_classification/ptask_multiclass_classification.py: Code for classification task.
  - hit.py: Code for prediction task.

## Train

To train the model in each dataset folder:

First please run deepwalk or other method as base embedding model, the embedding format is the same as deepwalk output.

```
python prep_dataset.py
python main.py
```
