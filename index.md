# Papers Classification and Visualization

## 1 Introduction

This lab focus on introducing the basic nlp methods and implementing them on a specific topic, to classify and visualize papers. In this lab, you will learn some basic concepts and steps in natural language processing and realize them using Python. Moreover, you will use Tensorflow to build your model and Tensorboard to visualize the results.

### 1.1 Documents Classification

Documents classification is a fundamental task in natural language processing. Traditional classification methods relies on feature engineering based on bag-of-words, production rules and linguistically-informed features. Since Mikolov opened word2vec source code in 2013 [1], the concept of embedding facilitates our classification task a lot. Recently, neural network based classification models greatly outperform previous linear models with hand-crafted sparse features. 

### 1.2 A Simple Classification Model

Firstly, we introduce FastText, which may be one of the simplest models that utilize word embeddings.

![](images/fasttext.jpeg)
Figure 1. Model architecture of FastText

FastText is used as the baseline of this lab. Figure 1 shows the architecture of FastText. An embedding lookup table is randomly initialized and updated through training.  Through embedding lookup layer, each word in a document is transferred to word embedding. The word representations are then averaged into a text representation, which is in turn fed to a linear classifier. More introductions of FastText can be found in [2].

### 1.3 Complex Classification Models

We introduce more complicated models here. n-gram rather than single word can be used in FastText. One dimensional cnn  [3] is a good tool for synthesizing meaning of a document. Rnn with attention mechanism [4] can also be used in classification. We can use pre-trained word representations [1] to initialize lookup table. 

### 1.4 Goals of this Lab

The basic requirements of this lab are:
- Use the given training set to train a basic FastText model; 
- Visualize trained word embeddings and paper embeddings;
- Write experiment report by LaTeX.

The bonus of this lab are:
- Choose one of the complex models or conceive your own model, implement it;
- Get an accuracy over 90%.

## 2 Get Ready

### 2.1 Installation of Python

You can download Python in https://www.python.org/downloads/.

### 2.2 Installation of TensorFlow and TensorBoard

You can refer to https://www.tensorflow.org/install/ for installation.

### 2.3 Installation of NLTK

NLTK is a natural language toolkit of python. Punkt is a package of NLTK and used to parse sentences. After installing NLTK, you can run python, import NLTK module and run nltk.download() to download needed NLTK packages.

### 2.4 Download Dataset and Example Code

In this experiment, we provide a training set, a validating set and a testing set. Testing set is only used to test trained models. Using testing set to control training procedure is considered as cheating.
Each set is in CSV form and self-explanatory. The testing set also has a TSV form, which is used as the metadata file for embedding visualization in TensorBoard.
You need to install python2 to run the example code.You need to upgrade TensorFlow and TensorBoard to their newest versions to run our example code. 
The dataset and example code can be downloaded here.




