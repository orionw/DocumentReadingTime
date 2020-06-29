import os
import glob
import json

import tqdm
import pandas as pd
import numpy as np
import argparse
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nltk.tokenize import word_tokenize


def create_story_files():
    if not os.path.isdir("ind_stories"):
        os.makedirs("ind_stories")
    stories = pd.read_csv("all_stories.tok", header=0, index_col=None, sep="\t")
    for story_num, story_df in stories.groupby("item"):
        story_words = " ".join(story_df.sort_values("zone").word.tolist())
        tokens = word_tokenize(story_words)
        with open(os.path.join("ind_stories", str(story_num) + ".txt"), "w") as fout:
            fout.write(" ".join(tokens))

def create_data_file_for_lmm():
    pd.options.mode.chained_assignment = None  # we're setting a slice of the DF new values, don't need this
    rt = pd.read_csv("processed_RTs.tsv", header=0, index_col=None, sep="\t")
    stories = pd.read_csv("all_stories.tok", header=0, index_col=None, sep="\t")
    matching_list = []
    for story_num, story_df in stories.groupby("item"):
        print("On story {}".format(story_num))
        model_output = pd.read_csv(os.path.join("ind_stories", "{}.output".format(story_num)), sep=" ", header=0, index_col=None)
        running_index = 0
        for index, (zone, zone_df) in enumerate(tqdm.tqdm(story_df.sort_values("zone").groupby("zone"))):
            # if it was tokenized, re-combine and sum (model needed tokenized, processed RTs from Natural Stories are not-tokenized)
            word = zone_df["word"].iloc[0]
            len_tokenized = len(word_tokenize(word)) if model_output.iloc[running_index, :]["word"] != word else 1
            potential_match = model_output.iloc[running_index:running_index+len_tokenized, :]

            if len(potential_match) > 1:
                potential_match = potential_match.sum()
            running_index += len_tokenized

            if type(potential_match) == pd.DataFrame:
                potential_match = potential_match.iloc[0]

            # double check the word
            assert potential_match["word"] == word or "<unk>" in potential_match["word"], "mismatched words: {} vs {}".format(potential_match["word"], word)
            potential_match["word"] = word

            # get the RT and join it together
            # item is the story, zone is the index in the story - all meanItemRTs should be the same, hence the mean
            rt_for_word = rt[(rt['word']==potential_match["word"]) & (rt['zone']==zone) & (rt['item']==story_num)]["meanItemRT"].mean() 
            assert not pd.isnull(rt_for_word), "got NA reading time"
            potential_match["RT"] = rt_for_word
            matching_list.append(potential_match.to_dict())
        
    full_df = pd.DataFrame(matching_list)
    full_df.to_csv("rt_for_lmm.csv")


def train_lmm():
    df = pd.read_csv("rt_for_lmm.csv", header=0, index_col=0)
    # following the appendix of https://arxiv.org/pdf/1808.09930.pdf minus the subject (since we have new subjects) and adaptation 
    md = smf.mixedlm("RT ~ wlen + sentpos + surp", df, groups=df["word"])
    mdf = md.fit()
    print("The summary for the fitted model is:\n", mdf.summary())

    # now gather the data from the documents and use that to predict the RT for each word, taking the sum
    # get the non-tokenized versions as our model expects that due to the natural stories processed RTs
    article_num_to_preds = {}
    for file_path in glob.glob(os.path.join("../data/article_texts/", "*.output")):
        article_num = file_path.split("/")[-1].replace(".output", "")
        cur_df = pd.read_csv(file_path, header=0, sep=" ", index_col=None)
        test_data = cur_df[["wlen", "sentpos", "surp", "word"]]
        rt_predictions = mdf.predict(exog=test_data)
        article_prediction = rt_predictions.sum() / 1000 # convert from MS to S
        article_num_to_preds[article_num] = article_prediction

    with open("../data/article_num_to_predictions_lmm.json", "w") as fout:
        json.dump(article_num_to_preds, fout)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create_story_files",
        action="store_true",
        default=False,
        help="Whether to split the tok files into individual files for the `neural_complexity` model",
    )
    parser.add_argument(
        "--create_data_file_for_lmm",
        action="store_true",
        default=False,
        help="Whether to join together the output from the `neural_complexity` model with the processed_RTs",
    )
    parser.add_argument(
        "--train_lmm",
        action="store_true",
        default=False,
        help="Whether to train the LMM on the processed RTs",
    )
    args = parser.parse_args()
    if args.create_story_files:
        create_story_files()
    if args.create_data_file_for_lmm:
        create_data_file_for_lmm()
    if args.train_lmm:
        train_lmm()