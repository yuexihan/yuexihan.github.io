# Papers Classification and Visualization

## Introduction

This lab focus on introducing the basic nlp methods and implementing them on a specific topic, to classify and visualize papers. In this lab, you will learn some basic concepts and steps in natural language processing and realize them using Python. Moreover, you will use Tensorflow to build your model and Tensorboard to visualize the results.

### Documents Classification

Documents classification is a fundamental task in natural language processing. Traditional classification methods relies on feature engineering based on bag-of-words, production rules and linguistically-informed features. Since Mikolov opened word2vec source code in 2013 [1], the concept of embedding facilitates our classification task a lot. Recently, neural network based classification models greatly outperform previous linear models with hand-crafted sparse features. 

### A Simple Classification Model

Firstly, we introduce FastText, which may be one of the simplest models that utilize word embeddings.
FastText is used as the baseline of this lab. Figure 1 shows the architecture of FastText. An embedding lookup table is randomly initialized and updated through training.  Through embedding lookup layer, each word in a document is transferred to word embedding. The word representations are then averaged into a text representation, which is in turn fed to a linear classifier. More introductions of FastText can be found in [2].

### Complex Classification Models

We introduce more complicated models here. n-gram rather than single word can be used in FastText. One dimensional cnn  [3] is a good tool for synthesizing meaning of a document. Rnn with attention mechanism [4] can also be used in classification. We can use pre-trained word representations [1] to initialize lookup table. 

### Goals of this Lab

The basic requirements of this lab are:
- Use the given training set to train a basic FastText model; 
- Visualize trained word embeddings and paper embeddings;
- Write experiment report by LaTeX.

The bonus of this lab are:
- Choose one of the complex models or conceive your own model, implement it;
- Get an accuracy over 90%.
