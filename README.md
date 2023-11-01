# FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks (NeurIPS 2023)
## Introduction

> This repository contains the implementation of the FedGCN algorithm, that leverages federated learning to efficiently train Graph Convolutional Network (GCN) models for semi-supervised node classification. It achieves rapid convergence while minimizing communication overhead. The algorithm implements a framework, where clients exclusively interact with the central server during a single pre-training step.

Paper: [FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks](https://arxiv.org/pdf/2201.12433.pdf)

## Installation

1. Clone the repository in your local machine from github directly or using the below command:

    `git clone https://github.com/yh-yao/FedGCN.git`

2. Running the code on a new virtual Python environment is recommended.

**2.1.	Create a new Python virtual environment:**

1. Install virtualenv on your system. Virtualenv is a tool to set up your Python environments.

    `pip install virtualenv` 

2. Create a new project folder, or go to the project folder where you have cloned the code in your terminal, and run the following command:

    `virtualenv <virtual-environment-name>`

3.	Now that a new Python virtual environment in created, activate it by using the below command in the same project folder as above:

    *On MacOS/Linux:*
    
    `source <virtual-environment-name>/bin/activate`

    *On Windows:*
    
    `venv\Scripts\activate`

You can also do the above steps using conda.

**2.2.	Setting up the environment:**

Run the below command to install all the packages listed in the requirements.txt file:

`pip install -r requirements.txt`

##	Running the code

### Local Simulation:

Once the code is in place along with all the required packages, the code can be run using the command below:

`python src/fedgcn_run.py -d=<dataset_name> -f=fedgcn -nl=<num_layers> -nhop=<num_hops> -iid_b=<beta_IID> -r=<repeat_frequency> -n=<num_trainers>`

For example:
`python src/fedgcn_run.py -d=cora -f=fedgcn -nl=2 -nhop=2 -iid_b=100 -r=3 -n=5`

You can also specify other arguments as needed. Here is a detailed list of all the arguments available:

| Argument | Overview | Data Type | Default value |
| -------- | -------- | -------- | -------- |
| -d | dataset_name | String | Cora |
| -f |	Fed_type |	String |	Fed_gcn |
| -nl |	Num_layers | Int |	2 |
| -nhop |	Num_hops |	Int |	2 |
| -iid_b |	Beta_IID |	Float |	10000 |
| -r |	Repeat_frequency |	Int	| 10 |
| -c |	num_global_rounds |	Int |	100 |
| -i |	num_local_step |	Int |	3 |
| -lr |	Learning_rate |	Float |	0.5 |
| -g |	If_gpu	|	
| -l |	Log_directory |	string |	./runs |
| -n |	Num_trainers |	int |	5 |


### Distributed Training:

1. Start the cluster, see `config.yaml` for dependency configuration
   ```shell
   $ ray up config.yaml
   ```
2. Submit enter-point script
   ```sell
   $ ray submit config.yaml fed_training.py 
   ```
3. Stop nodes in cluster, optionally you can terminate them using AWS console or CLI.
   ```shell
   $ ray down config.yaml
    ``` 

## Datasets

| Dataset |	Graph Type | #Nodes | #Edges | #Classes |
| -------- | -------- | -------- | -------- | -------- |
| Cora |	Citation Network |	2,708 |	10,556 | 7 |
| CiteSeer |	Citation Network |	3,327 |	9,104 |	6 |
| Reddit | Social Network| 232,965 | 114,615,892 | 41 |
| PubMed | Citation Network - Life Sciences | 19,717 |	88,648 | 3 |
| ogbn-products |	Product Recommendation | 2,449,029 | 61,859,140 | 47 |		
| ogbn-arxiv | Citation Network | 169,343 | 1,166,243 | 40 |
