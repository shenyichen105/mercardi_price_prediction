import numpy as np
from keras.regularizers import l2
from keras.optimizers import Adam
#define set of possible model parameters
model_params_selections = {
    "dropout_rate": [0.3, 0.4, 0.5, 0.6, 0.7],
    "r2_coeff": [0, 0.00005, 0.0001, 0.00015, 0.0002],
    "emb_size_text_desc": [20, 35, 40, 45, 50, 55 ,75, 100],
    "emb_size_text_name": [15, 20, 25, 30, 45, 55, 75],
    "emb_size_brand": [6,8,10,12],
    "emb_size_category": [4,6,8,10,12],
    "emb_size_condition": [2,4,6,8],
    "loss": ["mse"],
    "mask_zero": [False],
    "lr": [0.003, 0.0025,  0.002, 0.001],
    "metrics": [["mae"]],
    "kernel_size": [3,4,5],
    "n_filter": [10, 15, 20, 25],
    "fc_size_1": [32, 64, 128]
}

#define set of possible training parameters
training_params_selections = {
    "n_epochs": [15],
    "batch_size": [5000, 10000, 15000, 20000]
}

def generate_params_cnn():
    model_params = {}
    for key in model_params_selections:
        values = model_params_selections[key]
        index = np.random.choice(np.array(len(values)))
        model_params[key] = values[index]

    training_params = {}
    for key in training_params_selections:
        values = training_params_selections[key]
        index = np.random.choice(np.array(len(values)))
        training_params[key] = values[index]

    return model_params, training_params