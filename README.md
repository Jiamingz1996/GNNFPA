# GNN-FPA and GNN-SFPA
### Propagation is All You Need: A New Framework for Representation Learning and Classifier Training on Graphs

This repo contains the implementation of the model proposed in ["Propagation is All You Need: A New Framework for Representation Learning and Classifier Training on Graphs"](http://yangliang.github.io/pdf/mm23-zhuo.pdf).

**Installation**
To run the project you will need to install the required packages:

```
pip install -r requirements.txt 
```

**Dataset Source**

> All datasets are downloaded from package torch_geometric and saved as series of .pt file without any preprocess procedure. You can download the zipped dataset from release page of [torch_geometric](https://github.com/GitEventhandler/pytorch_geometric/tree/master/torch_geometric) and extract them to "./datasets" folder.

**Citation**

```
@article{zhuo2023propagation,
title={Propagation is All You Need: A New Framework for Representation Learning and Classifier Training on Graphs},
author={Zhuo, Jiaming and Cui, Can and Fu, Kun and Niu, Bingxin and He, Dongxiao and Guo, Yuanfang and Wang, Zhen and Wang, Chuan and Cao, Xiaochun and Yang, Liang},
year={2023},
booktitle = {{MM} '23: The {ACM} MULTIMEDIA Conference 2023},
}
