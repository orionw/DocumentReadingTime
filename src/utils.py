import os
import pandas as pd
import argparse
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize

SEED = 42 # int(os.environ['SEED']) if running multiple through the command line
print("Loaded seed is {}".format(SEED))


def set_seed(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def split_text_csv(data_path: str, output_dir: str = "."):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df = pd.read_csv(data_path, header=0, index_col=0)
    for index, (row) in df.iterrows():
        with open(os.path.join(output_dir, str(row["Survey_Num"]) + ".txt"), "w") as fout:
            fout.write(row["text"])
        with open(os.path.join(output_dir, str(row["Survey_Num"]) + "-tok.txt"), "w") as fout:
            fout.write(" ".join(word_tokenize(row["text"])))


def gather_surprisal_outputs(data_dir: str, output_path: str):
    article_num_to_surprisal = []
    # for the most accurate suprisal info from the model, needs tokenization
    for file_path in glob.glob(os.path.join(data_dir, "*.output-tok")):
        article_num = file_path.split("/")[-1].replace(".output-tok", "")
        cur_df = pd.read_csv(file_path, header=0, sep=" ", index_col=None)
        article_num_to_surprisal.append({"article_num": article_num, "surprisal": cur_df["surp"].sum()})
    
    print("Writing results to {}".format(output_path))
    pd.DataFrame(article_num_to_surprisal).to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_text",
        action="store_true",
        default=False,
        help="Whether to split the text into individual files for the `neural_complexity` model",
    )
    parser.add_argument(
        "--gather_suprisal",
        action="store_true",
        default=False,
        help="Whether to gather the output files from the `neural_complexity` model",
    )
    args = parser.parse_args()
    if args.split_text:
        split_text_csv(data_path="data/text_data.csv", output_dir="data/article_texts/")
    if args.gather_suprisal:
        gather_surprisal_outputs("data/article_texts", output_path="data/article_num_to_surprisal_only.csv")
    