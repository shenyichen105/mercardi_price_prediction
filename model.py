
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, LSTM, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
import math
from keras.optimizers import Adam

class model_base(object):
    def __init__(self, model_params, X, model_file=None):
        self.model = None
        #self.model_working_dir = working_dir
        self.model_params = model_params
        self.evaluate_results = None

        if model_file is None:
            self.define_model(X)
        else:
            self.load_model(model_file)

    def rnn_component(self, emb_item_desc, emb_name):
        """
        rnn component upon embeddings of item descriptin and item name
        """
        gru_hidden_size_name = self.model_params["gru_hidden_size_name"]
        gru_hidden_size_desc = self.model_params["gru_hidden_size_desc"]

        rnn_layer_desc = LSTM(gru_hidden_size_desc) (emb_item_desc)
        rnn_layer_name = LSTM(gru_hidden_size_name) (emb_name)

        return rnn_layer_desc, rnn_layer_name

    def fc_component(self, concatenate_features):

        """
        fully connected components using all information after extracted features of texts
        """

        dr_r = self.model_params["dropout_rate"]
        r2_reg_kernel = self.model_params["r2_reg_kernel"]

        #concatenate_features = BatchNormalization()(concatenate_features)
        fc = Dropout(dr_r) (Dense(64, kernel_regularizer=r2_reg_kernel) (concatenate_features))
        #fc = Dropout(dr_r) (Dense(64, kernel_regularizer=r2_reg_kernel) (concatenate_features))
        fc = Dropout(dr_r) (Dense(16, kernel_regularizer=r2_reg_kernel) (concatenate_features))

        return fc


    def _infer_input_size(self, X):

        """
        infer input/embedding sizes from data
        X: [X_train_name,  X_train_description, X_train_brand, X_train_category, \
            X_train_condition, X_train_shipping]
        """

        n_text = np.max([np.max(X[1]), np.max(X[0])]) + 1
        n_brand = np.max(X[2]) + 1
        n_category = np.max(X[3]) + 1
        n_condition = np.max(X[4]) + 1

        name_length = X[0].shape[1]
        desc_length = X[1].shape[1]

        return name_length, desc_length, n_text, n_brand, n_category, n_condition


    def define_model(self, X):

        """
        define a keras model here
        """
        #parameters

        emb_size_text_desc = self.model_params["emb_size_text_desc"]
        emb_size_text_name = self.model_params["emb_size_text_name"]
        emb_size_brand = self.model_params["emb_size_brand"]
        emb_size_category = self.model_params["emb_size_category"]
        emb_size_condition = self.model_params["emb_size_condition"]

        #input steps/embedding sizes
        name_length, desc_length, n_text, n_brand, n_category, n_condition = self._infer_input_size(X)

        #Inputs
        name = Input(shape=[name_length], name = "name")
        item_desc = Input(shape=[desc_length], name="item_desc")
        brand_name = Input(shape=[1], name="brand_name")
        category_name = Input(shape=[1], name="category_name")
        item_condition = Input(shape=[1], name="item_condition")
        shipping = Input(shape=[1], name="shipping")

        #Embeddings layers
        #word_emb = Embedding(n_text, emb_size_text, mask_zero = self.model_params["mask_zero"])
        emb_name = Embedding(n_text, emb_size_text_name, mask_zero = self.model_params["mask_zero"])(name)
        emb_item_desc = Embedding(n_text, emb_size_text_desc, mask_zero = self.model_params["mask_zero"])(item_desc)
        emb_brand_name = Embedding(n_brand, emb_size_brand)(brand_name)
        emb_category_name = Embedding(n_category, emb_size_category)(category_name)
        emb_item_condition = Embedding(n_condition, emb_size_condition)(item_condition)

        #rnn layer
        rnn_layer_desc, rnn_layer_name = self.rnn_component(emb_item_desc, emb_name)

        #main layer
        fc_features = concatenate([
                shipping
                , Flatten() (emb_brand_name)
                , Flatten() (emb_category_name)
                , Flatten() (emb_item_condition)
                , rnn_layer_desc
                , rnn_layer_name
            ])

        fc_features  = self.fc_component(fc_features)

        #output
        output = Dense(1, activation="linear")(fc_features)

        #model
        self.model = Model([name, item_desc, brand_name
                           , category_name, item_condition, shipping], output)

        print self.model.summary()

    def load_model(self, model_file):
        self.model = load_model(model_file)
        print "load model from model file: ", model_file

    def set_model_params(self, model_params):
        self.model_params = model_params

    def initialize_model(self):

        """
        compile keras model
        """

        loss = self.model_params["loss"]
        optimizer = self.model_params["optimizer"]
        metrics = self.model_params["metrics"]

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    @staticmethod
    def split_train_validation(X, y, train_percent = 0.9):

        """
        split training set into training and validation set
        """
        total_obs = X[0].shape[0]
        shuffled_idx = np.arange(total_obs)
        np.random.shuffle(shuffled_idx)

        train_idx = shuffled_idx[:int(total_obs * train_percent)]
        valid_idx = shuffled_idx[int(total_obs * train_percent):]

        X_train = [arr[train_idx] for arr in X]
        X_valid = [arr[valid_idx] for arr in X]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        return X_train, y_train, X_valid, y_valid

    def train_model(self, X_train, y_train, X_valid, y_valid, training_params, verbose=1):

        """
        train a keras model for x epochs
        input: X_train, X_valid : a list of features

        """
        n_epochs = training_params["n_epochs"]
        batch_size = training_params["batch_size"]

        self.model.fit(X_train, y_train, epochs=n_epochs, shuffle = True, batch_size=batch_size
          , validation_data=(X_valid, y_valid)
          , verbose=verbose)

    def save_model(self, outpath):
        self.model.save(outpath)

    def predict(self, X):
        return self.model.predict(X)

    def _rmsle(self, y, y_pred):
        assert len(y) == len(y_pred)
        to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
        return (sum(to_sum) * (1.0/len(y))) ** 0.5

    def predict_raw_price(self, X):
        raw_pred = self.model.predict(X, batch_size=2000, verbose=1)
        y_pred = np.exp(raw_pred) - 1
        y_pred = y_pred[:,0]
        return y_pred

    def evaluate(self, X, y_true):
        """
        evaluate results
        """
        y_pred = self.predict_raw_price(X)
        y_true = np.exp(y_true) - 1
        v_rmsle = self._rmsle(y_true, y_pred)
        print
        print "rmsle:" + str(v_rmsle)
        return v_rmsle


def load_and_pad_all_data(input_folder, max_name_len, max_des_len, test_set = False):
    """
    load all numpy data

    max_name_len: longest length of name considered
    max_des_len: longest length of item descriptin considered
    """
    #load
    X_id = np.load(input_folder + "X_id.npy")
    X_name = np.load(input_folder + "X_name.npy")
    X_category = np.load(input_folder + "X_category.npy")
    X_description = np.load(input_folder + "X_description.npy")
    X_shipping =np.load(input_folder + "X_shipping.npy")
    X_brand = np.load(input_folder + "X_brand.npy")
    X_condition = np.load(input_folder + "X_condition.npy")


    #pad sequences

    X_name = pad_sequences(X_name, maxlen=max_name_len)
    X_description = pad_sequences(X_description, maxlen=max_des_len)

    X = [X_name,  X_description, X_brand, X_category, X_condition, X_shipping]
    print "completed loading data!"

    if test_set:
        return X_id, X
    else:
        y = np.load(input_folder + "y_price.npy")
        return X_id, X, y

def main():

    input_folder = "../Data/Model_data/train/"
    X_id, X, y = load_and_pad_all_data(input_folder, max_name_len=None, max_des_len=100)

    #define model parameters
    model_params = {
        "dropout_rate": 0.5,
        "r2_reg_kernel": l2(0.0000),
        "emb_size_text_desc": 75,
        "emb_size_text_name": 25,
        "emb_size_brand": 16,
        "emb_size_category": 16,
        "emb_size_condition": 4,
        "gru_hidden_size_name": 16,
        "gru_hidden_size_desc": 32,
        "loss": "mse",
        #rnn mask zero should be true
        "mask_zero": True,
        "optimizer": Adam(lr = 0.003),
        "metrics": ["mae"]
    }
    #define training parameters
    training_params = {
        "n_epochs": 5,
        "batch_size": 30000
    }

    #model_file = "/Users/yichenshen/Documents/Mercardi_competition/Model/Saved_models/model1.h5"
    model_obj = model_base(model_params, X)
    X_train, y_train, X_valid, y_valid = model_base.split_train_validation(X, y)
    model_obj.initialize_model()
    model_obj.train_model( X_train, y_train, X_valid, y_valid, training_params)
    model_obj.evaluate(X_valid, y_valid)

    #generate submission
    # X_test_id, X_test = load_and_pad_all_data(input_folder, max_name_len = 21, max_des_len=100, test_set = True)
    # output_folder = "/Users/yichenshen/Documents/Mercardi_competition/Output/Output1"
    # preds = model_obj.predict_rmsle(X_test)

    # submission = pd.DataFrame({"test_id": X_test_id, "price": preds})
    # submission.to_csv(output_folder + "/myNNsubmission.csv", index=False)

if __name__ == "__main__": main()


