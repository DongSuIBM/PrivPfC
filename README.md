# PrivPfC: differentially private data publication for classification

PrivPfC is a differentially private algorithm for releasing data for classification analysis. The key idea underlying PrivPfC is to privately select, in a single step, a grid, which partitions the data domain into a number of cells. This selection is done by using the exponential mechanism with a novel quality function, which maximizes the expected number of correctly classified records by a histogram classifier. PrivPfC supports both the binary classification and the multiclass classification.  

For more details, please see our paper:
Dong Su, Jianneng Cao, Ninghui Li, Min Lyu: [PrivPfC: differentially private data publication for classification](https://link.springer.com/article/10.1007%2Fs00778-017-0492-3).  VLDB J. 27(2): 201-223 (2018).  

## How to run
- `cd ./src/single_grid`
- To publish the adult data for binary classification on the income attribute, 
  - `cd ./src/experiment_basic.py`
  - `python experiments_basic.py --dataset_name=adult --epsilon=1.0 --fold_num=0 --pool_size_threshold=10000`
- To publish the adult data for 3-class classification on the marital-status attribute, 
  - `cd ./src/experiment_basic.py`
  - `python experiments_basic.py --dataset_name=adult_marital-status --epsilon=1.0 --fold_num=0 --pool_size_threshold=10000`

## Environment requirements
- Python 2.7.12
- Numpy 1.12.1
- Pandas 0.19.1

Maintainer:
Dong Su, <sudong.tom@gmail.com>
