import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import TweetTokenizer, word_tokenize
from collections import Counter
from time import time

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
            print "no words found!"

        for key, value in self.word_dict.iteritems():
            if value > rare_threshold:
                self.words_idx[key] = idx
                idx +=1

        #add special words
        self.words_idx["special_rare"] = idx
        self.words_idx["special_missing"] = idx + 1
        self.total_words = idx

        print "new word index created."

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
            print "no tokenizer found!"
            return

        print "begin tokenizing names and items..."

        # create list of words from text
        self.splitted_names = self.data["name"].apply(lambda x: self.tokenizer.tokenize(unicode(x, "utf-8")))
        self.splitted_description = self.data["item_description"].apply(lambda x: [] if pd.isnull(x) else self.tokenizer.tokenize(unicode(x, "utf-8")))

        print "finished tokenizing"

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


    def preprocess(self, rare_thresholds=[60,5,4], words_idx=None):

        """
        a wrap up function for preprocess this data
        input:
        word_idx: external word index dictionary to serailize words
        rare_thresholds = [# of appearences threshold for text,
                           # of appearences threshold for categories,
                           # of appearences threshold for brand names]
        """

        print "started preprocess the datasets!"

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

        print "preprocess finished! time used: {:.2} s".format(end_time - start_time)

    def get_words_index(self):

        """
        output word index created
        """

        return self.words_idx


    def save_processed_data(self, output_folder):

        np.save(output_folder + "X_name", self.tokenized_names)
        np.save(output_folder + "X_description", self.tokenized_description)
        np.save(output_folder + "X_category", self.categories)
        np.save(output_folder + "X_brand", self.brand_names)
        np.save(output_folder + "X_condition", self.item_condition_id)
        np.save(output_folder + "X_id", self.item_id)
        np.save(output_folder + "X_shipping", self.shipping)

        if not self.is_test_data:
            np.save(output_folder + "y_price", self.log_price)

def main():

    data_folder = "../Data/Tsv_data/"
    train_file = "train.tsv"
    test_file = "test.tsv"

    train_data = pd.read_csv(data_folder + train_file, delimiter = "\t")
    test_data =  pd.read_csv(data_folder + test_file, delimiter = "\t")

    #parameters
    rare_thresholds = [30, 2, 2]

    #preprocess training data
    output_folder_train = "../Data/Model_data/train/"
    preprocess_train = preprocess_base(train_data)
    preprocess_train.preprocess(rare_thresholds)
    preprocess_train.save_processed_data(output_folder_train)

    train_words_idx = preprocess_train.get_words_index()

    del preprocess_train
    del train_data

    #preprocess test data
    output_folder_test = "../Data/Model_data/test/"
    preprocess_test = preprocess_base(test_data, is_test_data=True)
    preprocess_test.preprocess(rare_thresholds, words_idx=train_words_idx)
    preprocess_test.save_processed_data(output_folder_test)

    del preprocess_test
    del test_data

if __name__ == "__main__": main()




