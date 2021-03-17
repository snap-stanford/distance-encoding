# Distance-encoding for GNN design
This repository is the official PyTorch implementation of the *DEGNN* and *DEAGNN* framework reported in the paper: <br>
[*Distance-Encoding -- Design Provably More PowerfulGNNs for Structural Representation Learning*](https://arxiv.org/abs/2009.00142), to appear in NeurIPS 2020. 

The project's home page is: <http://snap.stanford.edu/distance-encoding/>

## Authors & Contact
Pan Li, Yanbang Wang, Hongwei Wang, Jure Leskovec

Questions on this repo can be emailed to <ywangdr@cs.stanford.edu> (Yanbang Wang)

## Installation
Requirements: Python >= 3.5, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate degnn-env
```

- Install PyTorch >= 1.4.0 and torch-geometric >= 1.5.0 (please refer to the [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) official websites for more details). Commands examples are:
```bash
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

The latest tested combination is: Python 3.8.2 + Pytorch 1.4.0 + torch-geometric 1.5.0.

## Quick Start
- To train **DEGNN-SPD** Task 2 (link prediction) on [C.elegans dataset](https://snap.stanford.edu/data/C-elegans-frontal.html): 
```bash
python main.py --dataset celegans --feature sp --hidden_features 100 --prop_depth 1 --test_ratio 0.1 --epoch 300
```
&nbsp;&nbsp;&nbsp; This uses 100-dimensional hidden features, 80/10/10 split of train/val/test set, and trains for 300 epochs.

- To train **DEAGNN-SPD** for Task 3 (node-triads prediction) on C.elegans dataset:
```bash
python main.py --dataset celegans_tri --hidden_features 100 --prop_depth 2 --epoch 300 --feature sp --max_sp 5 --l2 1e-3 --test_ratio 0.1 --seed 9
```
&nbsp;&nbsp;&nbsp; This enables 2-hop propagation per layer, truncates distance encoding at 5, and uses random seed 9.

- To train **DEGNN-LP** (i.e. the random walk variant) for Task 1 (node-level prediction) on usa-airports using average accuracy as evaluation metric:
```bash
python main.py --dataset usa-airports --metric acc --hidden_features 100 --feature rw --rw_depth 2 --epoch 500 --bs 128 --test_ratio 0.1
```

Note that here the `test_ratio` currently contains both validation set and the actual test set, and will be changed to contain only test set. 

- To generate **Figure2 LEFT** of the paper (Simulation to validate Theorem 3.3):
```bash
python main.py --dataset simulation --max_sp 10
```
&nbsp;&nbsp;&nbsp; The result will be plot to `./simulation_results.png`.


- All detailed training logs can be found at `<log_dir>/<dataset>/<training-time>.log`. A one-line summary will also be appended to `<log_dir>/result_summary.log` for each training instance.

## Usage Summary
```
Interface for DE-GNN framework [-h] [--dataset DATASET] [--test_ratio TEST_RATIO]
                                      [--model {DE-GNN,GIN,GCN,GraphSAGE,GAT}] [--layers LAYERS]
                                      [--hidden_features HIDDEN_FEATURES] [--metric {acc,auc}] [--seed SEED] [--gpu GPU]
                                      [--data_usage DATA_USAGE] [--directed DIRECTED] [--parallel] [--prop_depth PROP_DEPTH]
                                      [--use_degree USE_DEGREE] [--use_attributes USE_ATTRIBUTES] [--feature FEATURE]
                                      [--rw_depth RW_DEPTH] [--max_sp MAX_SP] [--epoch EPOCH] [--bs BS] [--lr LR]
                                      [--optimizer OPTIMIZER] [--l2 L2] [--dropout DROPOUT] [--k K] [--n [N [N ...]]]
                                      [--N N] [--T T] [--log_dir LOG_DIR] [--summary_file SUMMARY_FILE] [--debug]
```

## Optinal Arguments
```
  -h, --help            show this help message and exit
  
  # general settings
  --dataset DATASET     dataset name
  --test_ratio TEST_RATIO
                        ratio of the test against whole
  --model {DE-GCN,GIN,GAT,GCN,GraphSAGE}
                        model to use
  --layers LAYERS       largest number of layers
  --hidden_features HIDDEN_FEATURES
                        hidden dimension
  --metric {acc,auc}    metric for evaluating performance
  --seed SEED           seed to initialize all the random modules
  --gpu GPU             gpu id
  --adj_norm {asym,sym,None}
                        how to normalize adj
  --data_usage DATA_USAGE
                        use partial dataset
  --directed DIRECTED   (Currently unavailable) whether to treat the graph as directed
  --parallel            (Currently unavailable) whether to use multi cpu cores to prepare data
  
  # positional encoding settings
  --prop_depth PROP_DEPTH
                        propagation depth (number of hops) for one layer
  --use_degree USE_DEGREE
                        whether to use node degree as the initial feature
  --use_attributes USE_ATTRIBUTES
                        whether to use node attributes as the initial feature
  --feature FEATURE     distance encoding category: shortest path or random walk (landing probabilities)
  --rw_depth RW_DEPTH   random walk steps
  --max_sp MAX_SP       maximum distance to be encoded for shortest path feature
  
  # training settings
  --epoch EPOCH         number of epochs to train
  --bs BS               minibatch size
  --lr LR               learning rate
  --optimizer OPTIMIZER
                        optimizer to use
  --l2 L2               l2 regularization weight
  --dropout DROPOUT     dropout rate
  
  # imulation settings (valid only when dataset == 'simulation')
  --k K                 node degree (k) or synthetic k-regular graph
  --n [N [N ...]]       a list of number of nodes in each connected k-regular subgraph
  --N N                 total number of nodes in simultation
  --T T                 largest number of layers to be tested
  
  # logging
  --log_dir LOG_DIR     log directory
  --summary_file SUMMARY_FILE
                        brief summary of training result
  --debug               whether to use debug mode
```


## Reference
If you make use of the code/experiment of Distance-encoding in your work, please cite our paper:
```text
@article{li2020distance,
  title={Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning},
  author={Li, Pan and Wang, Yanbang and Wang, Hongwei and Leskovec, Jure},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```