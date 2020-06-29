import warnings
import sys
import os
import glob
import math
import json
import random

from flair.datasets import *
from flair.embeddings import *
from flair.data import *
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models.text_regression_model import TextRegressor
from flair.trainers import *

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformer import (train as transformer_train, evaluate as transformer_eval, main as transformer_main)
from mlp import MLPRegressor, fit_nn, predict_nn
from utils import SEED, set_seed

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

set_seed(SEED)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

class StandardModel:
    """
    The standard model for predicting reading time
    """
    def __init__(self, wpm=4):
        self.num_words_per_min = wpm

    def fit(self, **kwargs):
        return

    def predict(self, X_train):
        return self.wpm(X_train["Num_Words"])
    
    def wpm(self, words):
        # 4 words per second
        return words / self.num_words_per_min


class SurprisalModelLMM:
    """
    The surprisal model for predicting reading time
    Trained on Natural Stories reading time with an LMM
    Summed the word rt estimates
    """
    def __init__(self, predictions_dict):
        self.prediction_dict = predictions_dict

    def fit(self, **kwargs):
        return

    def predict(self, X_train):
        return X_train["Survey_Num"].apply(lambda x: self.prediction_dict[str(x)])


def get_scores(y_test, predictions, verbose=True):
    if verbose:
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, predictions))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(y_test, predictions))
    return math.sqrt(mean_squared_error(y_test, predictions)), mean_absolute_error(y_test, predictions)


def get_splits(train_index, test_index, df, y):
    curr_xtrain = df.iloc[train_index.tolist(), :]
    curr_xtest = df.iloc[test_index.tolist(), :]
    curr_ytrain = y.iloc[train_index.tolist()]
    curr_ytest = y.iloc[test_index.tolist()]
    return curr_xtrain, curr_xtest, curr_ytrain, curr_ytest


def create_mapping(df, embedder, shorten: bool = False):
    article_map = {}
    # creates a dict of the embeddings for easy lookup
    for num, article in enumerate(df["text"].unique()):
        sentence = Sentence(article if not shorten else " ".join(article.split(" ")[:300]))
        embedder.embed(sentence)
        embedding  = sentence.get_embedding()
        article_map[article] = embedding.detach().cpu().numpy()
    return article_map


def build_embedding_df(df, doc_embedder, embed_size=128, shorten: bool = False):
    data = np.zeros((df.shape[0], embed_size), dtype=float)
    article_map = create_mapping(df, doc_embedder, shorten=shorten)
    for index, article in enumerate(df["text"]):
        data[index, :] = article_map[article]

    new_df = pd.DataFrame(data.reshape(-1, embed_size))
    return new_df


def add_to_cv_dict(cv_scores, name, scores):
    cv_scores[name]["rmse"].append(scores[0])
    cv_scores[name]["mae"].append(scores[1])
    return


def run_cross_validation(model_path: str):
    full_results = {}
    current_location = os.getcwd()
    df = pd.read_csv(os.path.join("data", "screening_study_with_article_info.csv"), encoding="utf-8")
    text_df = pd.read_csv(os.path.join("data", "text_data.csv"), encoding="utf-8", index_col=0)
    y = df["Reading_Time"]
    df.drop("Reading_Time", axis=1, inplace=True)
    df.drop("Duration_Entire_Survey", axis=1, inplace=True)
    df = pd.get_dummies(df)

    full_train = df.copy(deep=True)
    full_train["Reading_Time"] = y
    full_train = full_train.merge(text_df, on="Survey_Num", how="outer")
    full_train = full_train[["text", "Reading_Time"]]
    text_only = full_train[["text"]]

    ### Gather surprisal baselines ###
    surprisal_only = pd.read_csv(os.path.join("data", "article_num_to_surprisal_only.csv"), header=0, index_col=0)
    surprisal_only_dict = dict(zip(surprisal_only.article_num, surprisal_only.surprisal))
    # a function to map the article num to the surprisal sum
    def data_to_surprisal_only(train_data):
        return train_data["Survey_Num"].apply(lambda x: surprisal_only_dict[x]).to_numpy().reshape(-1, 1)
    # gather the summed RTs from the LMM
    with open(os.path.join("data", "article_num_to_predictions_lmm.json"), "r") as fin:
        lmm_surprsial = json.load(fin)

    cv = KFold(n_splits=10, random_state=SEED, shuffle=False)

    data_only_cv = {
        "rf": {"rmse": [], "mae": []},
        "knn": {"rmse": [], "mae": []},
        "mlp": {"rmse": [], "mae": []},
        "lr": {"rmse": [], "mae": []},
        "lr-basic": {"rmse": [], "mae": []},
        "std": {"rmse": [], "mae": []},
        "lmm": {"rmse": [], "mae": []},
        "lr-surp": {"rmse": [], "mae": []},
    }

    """
    Data Only Models:
    includes:
        MLP Regressor
        Random Forest
        Linear Regression (word-only, regular, surprisal-only)
        Standard 240 WPM Model
        KNN
        RT LMM model (used by dict params from previous train)
    """
    print("Training data only models ...")
    for index, (train_index, test_index) in enumerate(cv.split(df)):
        print("On training split {}".format(index + 1))
        # instantiate models
        mlp = MLPRegressor(df.shape[1], 100, 1)
        rf = RandomForestRegressor(random_state=SEED)
        knn = KNeighborsRegressor()
        lr = LinearRegression()
        lr_surp = LinearRegression() # truly awful, not enough info to do good
        basic_lr = LinearRegression()
        std = StandardModel()
        surprisal_lmm = SurprisalModelLMM(lmm_surprsial)

        # set up datasets
        curr_xtrain = df.iloc[train_index.tolist(), :]
        curr_xtest = df.iloc[test_index.tolist(), :]
        curr_ytrain = y.iloc[train_index.tolist()]
        curr_ytest = y.iloc[test_index.tolist()]

        # fit
        mlp = fit_nn(mlp, curr_xtrain, curr_ytrain, n_epochs=100)
        rf.fit(curr_xtrain, curr_ytrain)
        knn.fit(curr_xtrain, curr_ytrain)
        lr.fit(curr_xtrain, curr_ytrain)
        basic_lr.fit(curr_xtrain[["Num_Words"]], curr_ytrain)
        lr_surp.fit(data_to_surprisal_only(curr_xtrain), curr_ytrain)

        # # calculate scores
        pred_mlp = predict_nn(mlp, curr_xtest)
        scores = get_scores(curr_ytest, pred_mlp, verbose=False)
        add_to_cv_dict(data_only_cv, "mlp", scores)

        pred_rf = rf.predict(curr_xtest)
        scores = get_scores(curr_ytest, pred_rf, verbose=False)
        add_to_cv_dict(data_only_cv, "rf", scores)

        pred_knn = knn.predict(curr_xtest)
        scores = get_scores(curr_ytest, pred_knn, verbose=False)
        add_to_cv_dict(data_only_cv, "knn", scores)

        pred_lr = lr.predict(curr_xtest)
        scores = get_scores(curr_ytest, pred_lr, verbose=False)
        add_to_cv_dict(data_only_cv, "lr", scores)

        pred_lr_basic = basic_lr.predict(curr_xtest[["Num_Words"]])
        scores = get_scores(curr_ytest, pred_lr_basic, verbose=False)
        add_to_cv_dict(data_only_cv, "lr-basic", scores)

        pred_lr_surp = basic_lr.predict(data_to_surprisal_only(curr_xtest))
        scores = get_scores(curr_ytest, pred_lr_surp, verbose=False)
        add_to_cv_dict(data_only_cv, "lr-surp", scores)

        pred_std = std.predict(curr_xtest)
        scores = get_scores(curr_ytest, pred_std, verbose=False)
        add_to_cv_dict(data_only_cv, "std", scores)

        pred_lmm = surprisal_lmm.predict(curr_xtest)
        scores = get_scores(curr_ytest, pred_lmm, verbose=False)
        add_to_cv_dict(data_only_cv, "lmm", scores)

    # print out extracted data only reports
    for key, value in data_only_cv.items():
        for metric, scores in value.items():
            print('The model {} got an average of {} for {}'.format(key, np.mean(scores), metric))


    for model_dir in glob.glob(os.path.join(model_path, "*")):
        model_name = model_dir.split("/")[-1]
        print("Evaluating model", model_name)
        text_only_cv = {
            f"{model_name}": {"rmse": [], "mae": []},
        }

        stacked_cv = {
            f"{model_name}/MLP": {"rmse": [], "mae": []},
        }

        trained_model = TextRegressor.load(os.path.join(model_dir, "best-model.pt"))
        doc_embedder = trained_model.document_embeddings
        sentence = Sentence('The grass is green . And the sky is blue .')
        doc_embedder.embed(sentence)
        test_embed  = sentence.get_embedding()
        EMBEDDING_SIZE = test_embed.shape[0]
        assert test_embed is not None, "embedded a None object"

        # build combined text and embedding vector
        embedding_df = build_embedding_df(text_only, doc_embedder, embed_size=EMBEDDING_SIZE, shorten=True if model_name == "roBERTa" else False)
        combined_df = pd.concat([df, embedding_df], axis=1)
        assert combined_df.shape[0] == df.shape[0] and combined_df.shape[1] == df.shape[1] + embedding_df.shape[1], \
                "shapes were not aligned: df {} combined {}, embed {}".format(df.shape, combined_df.shape, embedding_df.shape)

        """
        Text Only (Embedding) Models:
        includes:
            LSTM
            Transformers
        # """
        print("Training text only models ...")
        for train_index, test_index in cv.split(df):
            # instantiate models
            mlp = MLPRegressor(test_embed.shape[0], 100, 1)
            # set up datasets and combine with text
            curr_xtrain, curr_xtest, curr_ytrain, curr_ytest = get_splits(train_index, test_index, embedding_df, y)
            # fit
            mlp = fit_nn(mlp, curr_xtrain, curr_ytrain, n_epochs=100)

            # calculate scores
            pred_mlp = predict_nn(mlp, curr_xtest)
            scores = get_scores(curr_ytest, pred_mlp, verbose=False)
            add_to_cv_dict(text_only_cv, model_name, scores)

        # print out extracted data only reports
        for key, value in text_only_cv.items():
            for metric, scores in value.items():
                print('The model {} got an average of {} for {}'.format(key, np.mean(scores), metric))


        """
        Stacked Models:
        includes:
            LSTM / MLP Regressor
            Transformers / MLP Regressor
        """
        print("Training stacked models ...")
        for train_index, test_index in cv.split(df):
            # instantiate models
            mlp = MLPRegressor(df.shape[1] + test_embed.shape[0], 100, 1)
            # set up datasets and combine with text
            curr_xtrain, curr_xtest, curr_ytrain, curr_ytest = get_splits(train_index, test_index, combined_df, y)

            # fit
            mlp = fit_nn(mlp, curr_xtrain, curr_ytrain, n_epochs=100)

            # calculate scores
            pred_mlp = predict_nn(mlp, curr_xtest)
            scores = get_scores(curr_ytest, pred_mlp, verbose=False)
            add_to_cv_dict(stacked_cv, model_name + "/MLP", scores)

        # print out extracted data only reports
        for key, value in stacked_cv.items():
            for metric, scores in value.items():
                print('The model {} got an average of {} for {}'.format(key, np.mean(scores), metric))

        full_results = {**full_results, **stacked_cv, **text_only_cv}


    #### create overall dataframe from results ####
    results = pd.DataFrame(columns=["Group", "Name", "MAE", "MAE-STD", "RMSE", "RMSE-STD"])
    for (group_name, dict_results) in [("Data-Only", data_only_cv), ("Full_Results", full_results)]:
        for key, value in dict_results.items():
            results = results.append(pd.Series({"Group": group_name, "Name": key, "MAE": np.mean(value["mae"]),
                             "RMSE": np.mean(value["rmse"]), "RMSE-STD": np.std(value["rmse"]),
                             "MAE-STD": np.std(value["mae"])}), ignore_index=True)
    assert results.shape[0] != 0, "no data added to the dataframe"
    if not os.path.isdir("results"):
        os.makedirs("results")
    results.to_csv("results/all_results-{}-{}.csv".format(SEED, model_path.split("/")[0]))

if __name__ == "__main__":
    # use the path to a folder containing the trained document embedders (see `document_embedder.oy`)
    run_cross_validation("rnn_trained/") 
