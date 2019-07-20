# LEAM
This repository contains source code necessary to reproduce the results presented in the paper [Joint Embedding of Words and Labels for Text Classification](https://arxiv.org/pdf/1805.04174.pdf) (ACL 2018):

```
@inproceedings{wang_id_2018_ACL,
  title={Joint Embedding of Words and Labels for Text Classification},
  author={Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen, Xinyuan Zhang, Ricardo Henao, Lawrence Carin},
  booktitle={ACL},
  year={2018}
}
```



Comparison Illustration of proposed **LEAM** with traditional methods for text sequence representations

Traditional Method           |  LEAM: Label Embedding Attentive Model
:-------------------------:|:-------------------------:
![](/plots/schemes/scheme_a.png) |  ![](/plots/schemes/scheme_b.png)
Directly aggregating word embedding **V** for text sequence representation **z** | We leverage the “compatibility” **G** between embedded words **V** and labels **C** to derive the attention score **β** for improved **z**.




## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies)
2. [Prepare datasets](#prepare-datasets)
3. [Training](#training)
    1. Training on standard dataset
    2. Training on your own dataset
4. [Reproduce paper figure results](#reproduce-paper-figure-results)

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/). Additional dependencies for running experiments are: `numpy`, `cPickle`, `scipy`, `math`, `gensim`. 

## Prepare datasets

We consider the following datasets: Yahoo, AGnews, DBPedia, yelp, yelp binary. For convenience, we provide pre-processed versions of all datasets. Data are prepared in pickle format. Each `.p` file has the same fields in same order: `train text`, `val text`, `test text`, `train label`, `val label`, `test label`, `dictionary` and `reverse dictionary`.


To run your own dataset, please follow the code in `preprocess.py` to put your own dataset under `data` directory. The default dataset is on `mbu.csv`. 

## Training
**1. Training on standard dataset**

To run the test, use the command `python model.py`. 

## Prediction and visuliation on 
Ten topics prediction from yahoo

![](/plots/schemes/predictions.png) 




