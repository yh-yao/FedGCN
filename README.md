FedGCN
=====

This repository contains the code for [FedGCN: Convergence and Communication Tradeoffs in Federated Training of Graph Convolutional Networks](https://arxiv.org/abs/2201.12433).


- Current implementation of FedGCN assumes known adj matrix and node features of L-hop neighbors to avoid implementation of communication (computational equavelient). 
- Open to questions and collaborations. yuhangya@andrew.cmu.edu
- Keep updating. Will support more datasets, compare algorithms, and (real) distributed training soon.


## Data

4 datasets were used in the paper:

- Cora
- Citeseer
- Pubmed
- Ogbn-ArXiv

## Requirements
  * Python 3
  * PyTorch
  * networkx
  * numpy
  * torch-geometric

## Main file
  * main.ipynb
  * Theory Evaluation.ipynb
  * Communication Cost.ipynb
