# Pytorch Sentiment Classification
### on A clean and simple template by  FrancescoSaverioZuppichini🚀🚀
*Dxzmpk*

In this article, we present you a deep learning project called Pytorch Sentiment Classification.
I will use the method proposed in <<Kim Y. Convolutional Neural Networks for Sentence Classification. 2014>>

- modularity: we split each logic piece into a different python submodule
- ready to go: by using [poutyne](https://pypi.org/project/Poutyne/) a Keras-like framework you don't have to write any train loop.
- [torchsummary](https://github.com/sksq96/pytorch-summary) to show a summary of your models
- reduce the learning rate on a plateau
- auto-saving the best model
- experiment tracking with [comet](https://www.comet.ml/)
- logging using python [logging](https://docs.python.org/3/library/logging.html) module
## Installation
Clone the repo and go inside it. Then, run:

```
pip install -r requirements.txt
```

## Run

```
python train.py
```
don't forget to download nltk_data for yourself, or the program will report an error 


## Train/Evaluation
In our case we kept things simple, all the training and evaluation logic is inside `.main.py` where we used [poutyne](https://pypi.org/project/Poutyne/) as the main library. We already defined a useful list of callbacks:
- learning rate scheduler
- auto-save of the best model
- early stopping
Usually, this is all you need!
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/develop/images/main.png?raw=true)
### Callbacks 
You may need to create custom callbacks, with [poutyne](https://pypi.org/project/Poutyne/) is very easy since it support Keras-like API. You custom callbacks should go inside `./callbacks`. For example, we have created one to update Comet every epoch.
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/develop/images/CometCallback.png?raw=true)

### Track your experiment
We are using [comet](https://www.comet.ml/) to automatically track our models' results. This is what comet's board looks like after a few models run.
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/develop/images/comet.jpg?raw=true)
Running `main.py` produces the following output:
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/develop/images/output.jpg?raw=true)

## Conclusions
I hope you found some useful information and hopefully it this template will help you on your next amazing project :)

