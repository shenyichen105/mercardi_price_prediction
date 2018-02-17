from model import model_base
from model import load_and_pad_all_data
from keras.layers import Conv1D, MaxPooling1D, Concatenate, GlobalMaxPooling1D, BatchNormalization, Flatten, Dropout, Dense
from keras.regularizers import l1, l2
import math
from keras.optimizers import Adam
import numpy as np
import json
from parameter_search import generate_params_cnn
import argparse


class model_cnn(model_base):
    def __init__(self, model_params, X, model_file=None):
         model_base.__init__(self, model_params, X, model_file)

    def rnn_component(self, emb_item_desc, emb_name):
        """
        rnn/cnn component

        using embeddings of item descriptions and item names
        """
        r2_reg_kernel = self.model_params["r2_reg_kernel"]
        dr_r = self.model_params["dropout_rate"]

        kernel_size = self.model_params["kernel_size"]
        n_filter = self.model_params["n_filter"]

        cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(emb_item_desc)
        #cnn = BatchNormalization()(cnn)
        # cnn = Conv1D(4, 3, activation="relu", kernel_regularizer=r2_reg_kernel)(cnn)
        # #cnn = BatchNormalization()(cnn)
        #cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(cnn)
        #cnn = MaxPooling1D(2)(cnn)
        cnn = Dropout(dr_r)(Flatten()(cnn))
        rnn_layer_desc = cnn


        cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(emb_name)
        #cnn = BatchNormalization()(cnn)
        # cnn = Conv1D(4, 3, activation="relu", kernel_regularizer=r2_reg_kernel)(cnn)
        # #cnn = BatchNormalization()(cnn)
        #cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(cnn)
        #cnn = MaxPooling1D(2)(cnn)
        cnn = Dropout(dr_r)(Flatten()(cnn))
        rnn_layer_name = cnn

        return rnn_layer_desc, rnn_layer_name



    def fc_component(self, concatenate_features):

        """
        fully connected components using all information after extracted features of texts
        """

        dr_r = self.model_params["dropout_rate"]
        r2_reg_kernel = self.model_params["r2_reg_kernel"]
        fc_size_1 = self.model_params["fc_size_1"]

        #concatenate_features = BatchNormalization()(concatenate_features)
        fc = Dropout(dr_r) (Dense(fc_size_1, kernel_regularizer=r2_reg_kernel) (concatenate_features))
        #fc = Dropout(dr_r) (Dense(32, kernel_regularizer=r2_reg_kernel) (fc))

        return fc

def main():

    #parse the command line arguments "-t" or "-tuning"
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning', '-t',action='store_true', help='parameter tuning flag')
    args = parser.parse_args()


    input_folder = "../Data/Model_data/train/"
    X_id, X, y = load_and_pad_all_data(input_folder, max_name_len=None, max_des_len=70)

    print "completed loading data...."
    print

    #define model parameters
    model_params = {
        "dropout_rate": 0.3,
        "r2_coeff": 0.0001,
        "emb_size_text_desc": 45,
        "emb_size_text_name": 55,
        "emb_size_brand": 8,
        "emb_size_category": 6,
        "emb_size_condition": 2,
        "loss": "mse",
        "mask_zero": False,
        "lr": 0.003,
        "metrics": ["mae"],
        "kernel_size": 5,
        "n_filter": 20,
        "fc_size_1": 64
    }
    #define training parameters
    training_params = {
        "n_epochs": 15,
        "batch_size": 15000
    }

    #model_file = "/Users/yichenshen/Documents/Mercardi_competition/Model/Saved_models/model1.h5"
    def run(model_params, training_params):
        #log model params
        with open('result.txt', 'a') as f:
            f.write(json.dumps(model_params) + '\n')
            f.write(json.dumps(training_params) + '\n')

        model_params["optimizer"] = Adam(lr=model_params["lr"])
        model_params["r2_reg_kernel"] = l2(model_params["r2_coeff"])

        model_obj = model_cnn(model_params, X)
        X_train, y_train, X_valid, y_valid = model_cnn.split_train_validation(X, y)

        #start training model
        model_obj.initialize_model()
        print "start training..."
        print
        model_obj.train_model( X_train, y_train, X_valid, y_valid, training_params)
        result = model_obj.evaluate(X_valid, y_valid)
        #log results into a file
        with open('result.txt', 'a') as f:
            f.write('rsmle:' + str(result)+ '\n')
            f.write('\n')
        print "finished training"

    #parameter tuning mode
    if args.tuning:
        with open('result.txt', 'w') as f:
            f.write("started parameters searchings....." + '\n')
        n_iter = 50
        #random grid search
        for i in range(n_iter):
            model_params, training_params = generate_params_cnn()
            run(model_params, training_params)
    #normal mode
    else:
        run(model_params, training_params)

if __name__ == "__main__": main()