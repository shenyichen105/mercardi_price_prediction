import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import TweetTokenizer, word_tokenize
from collections import Counter
from time import time

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, Activation, concatenate, Embedding
from keras.layers import Conv1D, MaxPooling1D, Concatenate, BatchNormalization, Flatten, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
import math
from keras.optimizers import Adam

class preprocess_base(object):

    def __init__(self, data, tokenizer=None, is_test_data=False):
        """take the raw dataframe"""
        self.data = data
        self.is_test_data = is_test_data
        #default tokenizer
        if tokenizer is None:
            self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
        else:
            self.tokenizer = tokenizer
        #label encoder
        self.label_encoder = LabelEncoder()

        #bag of words
        self.word_dict = Counter()
        self.words_idx = {}
        self.total_words = 0

        #processed data
        self.splitted_names = None
        self.splitted_description = None
        self.tokenized_names = None
        self.tokenized_description = None
        self.categories = None
        self.brand_names = None
        self.item_condition_id = None
        self.item_id = None
        self.log_price = None


    def replace_speical_words(self):

        """
        remove special character
        """

        self.data["name"] = self.data["name"].str.replace("\[rm\]", "special_rm")
        self.data["item_description"] = self.data["item_description"].str.replace("\[rm\]", "special_rm")

    def _add_words_to_dict(self, wlist):

        """
        helper funtion to add word to word dict
        """

        for w in wlist:
            self.word_dict[w] +=1

    def create_word_index(self, rare_threshold=50):

        """
        create index for each word appeared greater than a threshold
        for rare words =, substitute with special_rare
        for missing , substitute with  special_missing
        """

        #adding words to a word dictionary
        _ = self.splitted_names.apply(lambda x: self._add_words_to_dict(x))
        _ = self.splitted_description.apply(lambda x: self._add_words_to_dict(x))

        idx = 1
        if self.word_dict is None:
            print("no words found!")

        for key, value in self.word_dict.items():
            if value > rare_threshold:
                self.words_idx[key] = idx
                idx +=1

        #add special words
        self.words_idx["special_rare"] = idx
        self.words_idx["special_missing"] = idx + 1
        self.total_words = idx

        print("new word index created.")

    def _serialize_word_list(self, x):

        """
        helper funtion to index words
        input x: a list of words
        output: a list of index based on word index dictionary
        """

        if len(x) == 0:
            return [self.words_idx["special_missing"]]
        else:
            return [self.words_idx[w] if w in self.words_idx else self.words_idx["special_rare"] for w in x]


    def tokenize_text(self, rare_threshold=50):

        """
        tokenize name and description
        """

        if self.tokenizer is None:
            print("no tokenizer found!")
            return

        print("begin tokenizing names and items...")

        # create list of words from text
        self.splitted_names = self.data["name"].apply(lambda x: self.tokenizer.tokenize(x))
        self.splitted_description = self.data["item_description"].apply(lambda x: [] if pd.isnull(x) else self.tokenizer.tokenize(x))

        print("finished tokenizing")

    def serialize_text(self):

        """
        serialize word list
        """

        self.tokenized_names =  self.splitted_names.apply(lambda x: self._serialize_word_list(x))
        self.tokenized_description =  self.splitted_description.apply(lambda x: self._serialize_word_list(x))


    def _get_rare_items_from_column(self, column, threshold):
        counts = column.value_counts()
        rare_items_list = counts[counts < threshold].index.values
        return rare_items_list

    def serialize_category(self, rare_threshold=5):

        """
        serailize categories
        """

        categories = self.data["category_name"].copy()
        #missing value replaced by special token
        categories[categories.isnull()] = "special_missing"
        #rare categories replaced by special token
        rare_cates = self._get_rare_items_from_column(categories, rare_threshold)
        categories[categories.isin(rare_cates)] = "special_rare"
        #serialze
        self.categories  = self.label_encoder.fit_transform(categories)

    def serialize_brand_name(self, rare_threshold=3):

        """
        serailize brandnames
        """

        brand_names = self.data["brand_name"].copy()
        #missing value replaced by special token
        brand_names[brand_names.isnull()] = "special_missing"
        #rare categories replaced by special token
        rare_brands = self._get_rare_items_from_column(brand_names, rare_threshold)
        brand_names[brand_names.isin(rare_brands)] = "special_rare"
        #serialize
        self.brand_names  = self.label_encoder.fit_transform(brand_names)

    def process_price(self):

        """
        process price data
        """

        self.log_price = np.log(self.data["price"].values + 1)

    def process_others(self):

        """
        process item condition id, item id, shipping
        """

        self.item_condition_id = self.data["item_condition_id"].values -1

        if not self.is_test_data:
            self.item_id = self.data["train_id"].values
        else:
            self.item_id = self.data["test_id"].values

        self.shipping = self.data["shipping"].values


    def preprocess(self, rare_thresholds=[50,5,3], words_idx=None):

        """
        a wrap up function for preprocess this data
        input:
        word_idx: external word index dictionary to serailize words
        rare_thresholds = [# of appearences threshold for text,
                           # of appearences threshold for categories,
                           # of appearences threshold for brand names]
        """

        print("started preprocess the datasets!")

        start_time = time()

        self.replace_speical_words()
        self.tokenize_text()

        if words_idx is None:
            self.create_word_index(rare_threshold=rare_thresholds[0])
        else:
            self.words_idx = words_idx

        self.serialize_text()
        self.serialize_category(rare_threshold=rare_thresholds[1])
        self.serialize_brand_name(rare_threshold=rare_thresholds[2])

        if not self.is_test_data:
            self.process_price()

        self.process_others()

        end_time  = time()

        print("preprocess finished! time used: {:.2f} s".format(end_time - start_time))

    def get_words_index(self):

        """
        output word index created
        """

        return self.words_idx


    def get_processed_data(self):

        processed_data = {}
        processed_data["X_id"] = self.item_id
        processed_data["X_name"] = self.tokenized_names
        processed_data["X_description"] = self.tokenized_description
        processed_data["X_category"] = self.categories
        processed_data["X_brand"] = self.brand_names
        processed_data["X_condition"] = self.item_condition_id
        processed_data["X_shipping"] = self.shipping

        if not self.is_test_data:
            processed_data["y_price"]  = self.log_price

        return processed_data


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
        rnn/cnn component

        using embeddings of item descriptions and item names
        """
        r2_reg_kernel = self.model_params["r2_reg_kernel"]
        dr_r = self.model_params["dropout_rate"]

        kernel_size = self.model_params["kernel_size"]
        n_filter = self.model_params["n_filter"]

        cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(emb_item_desc)
        cnn = Dropout(dr_r)(Flatten()(cnn))
        rnn_layer_desc = cnn


        cnn = Conv1D(n_filter, kernel_size, activation="relu", kernel_regularizer=r2_reg_kernel)(emb_name)
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

        fc = Dropout(dr_r) (Dense(fc_size_1, kernel_regularizer=r2_reg_kernel) (concatenate_features))

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

        print(self.model.summary())

    def load_model(self, model_file):
        self.model = load_model(model_file)
        print("load model from model file: " + model_file)

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

    def train_model(self, X_train, y_train, X_valid, y_valid, training_params):

        """
        train a keras model for x epochs
        input: X_train, X_valid : a list of features

        """
        n_epochs = training_params["n_epochs"]
        batch_size = training_params["batch_size"]

        self.model.fit(X_train, y_train, epochs=n_epochs, shuffle = True, batch_size=batch_size
          , validation_data=(X_valid, y_valid)
          , verbose=1)

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
        print("rmsle:" + str(v_rmsle))

def pad_text_seq_data(processed_data, max_name_len, max_des_len, test_set = False):
    """
    max_name_len: longest length of name considered
    max_des_len: longest length of item descriptin considered
    """
    #pad sequences
    processed_data["X_name"]= pad_sequences(processed_data["X_name"], maxlen=max_name_len)
    processed_data["X_description"] = pad_sequences(processed_data["X_description"], maxlen=max_des_len)

    X_id = processed_data["X_id"]
    X = [processed_data["X_name"],  processed_data["X_description"]
        , processed_data["X_brand"], processed_data["X_category"]
        , processed_data["X_condition"], processed_data["X_shipping"]]

    print("completed padding data!")
    if test_set:
        return X_id, X
    else:
        y = processed_data["y_price"]
        return X_id, X, y

def main():
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    # from subprocess import check_output
    # print(check_output(["ls", "../input"]).decode("utf8"))

    # Any results you write to the current directory are saved as output.
    start_time = time()
    print("Loading data...")

    data_folder = "../Data/Tsv_data/"
    train_file = "train.tsv"
    test_file = "test.tsv"

    train_data = pd.read_csv(data_folder + train_file, delimiter = "\t")
    test_data =  pd.read_csv(data_folder + test_file, delimiter = "\t")

    # train = pd.read_table("../input/train.tsv")
    # test = pd.read_table("../input/test.tsv")

    # print(train.shape)
    # print(test.shape)
    print("finished loading data!")

    #parameters
    rare_thresholds = [30, 2, 2]

    #preprocess training data
    preprocess_train = preprocess_base(train_data)
    preprocess_train.preprocess(rare_thresholds)
    processed_data_train = preprocess_train.get_processed_data()
    X_train_id, X_train, y_train = pad_text_seq_data(processed_data_train
                                            ,max_name_len = 21, max_des_len=70)


    train_words_idx = preprocess_train.get_words_index()

    #preprocess testing data
    preprocess_test = preprocess_base(test_data, is_test_data=True)
    preprocess_test.preprocess(rare_thresholds, words_idx=train_words_idx)
    processed_data_test = preprocess_test.get_processed_data()
    X_test_id, X_test = pad_text_seq_data(processed_data_test, max_name_len = 21
                                          , max_des_len=70, test_set=True)

    #train model
    #define model parameters
    model_params = {
        "dropout_rate": 0.3,
        "r2_reg_kernel": l2(0.0001),
        "emb_size_text_desc": 45,
        "emb_size_text_name": 55,
        "emb_size_brand": 8,
        "emb_size_category": 6,
        "emb_size_condition": 2,
        "loss": "mse",
        "mask_zero": False,
        "optimizer": Adam(lr=0.003),
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

    #train model
    model_obj = model_base(model_params, X_train)
    model_obj.initialize_model()
    X_train_1, y_train_1, X_valid, y_valid = model_base.split_train_validation(X_train, y_train)
    model_obj.train_model(X_train_1, y_train_1, X_valid, y_valid, training_params)

    end_time = time()
    print("total time used:{:.2f}s".format(start_time - end_time))
    # # #evaluate model
    # model_obj.evaluate(X_valid, y_valid)
    # end_time = time()
    # print("grand total time used: {:.2f}s".format(end_time - start_time))
    y_preds = model_obj.predict_raw_price(X_test)
    submission = pd.DataFrame({"test_id": X_test_id, "price": y_preds})
    submission.to_csv("myNNsubmission.csv", index=False)

if __name__ == "__main__": main()



