Mercardi Price Suggestion Challenge

https://www.kaggle.com/c/mercari-price-suggestion-challenge

Used word embedding + RNN/CNN to predict prices from raw features and text
These pieces of code serve as an basic ML framework that modularizes each part of the ML pipline in a scalable fashion: preprocessing module, model components, parameter tuning/grid search, which can be easily extended to other ML tasks.

preprocess.py -- base preprocessing module

model.py -- base model class (RNN), with a main function to run the model

cnn_based_model.py -- CNN model inhered from base model class, with a main function to run the model

parameter_search.py -- grid search

