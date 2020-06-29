## Original Implementation and Dataset of the Paper ['You Don’t Have Time to Read This: an Exploration of Document-Level Reading Time Prediction'](https://www.aclweb.org/anthology/2020.acl-main.162/) by Weller et. al. published at ACL 2020.  

## Datasets:
Datasets can be found in `src/data`.  If you want all the data, use the `src/data/screening_study_with_article_info.csv` file which contains the article texts and the demographic info.  The other data files are explained in the readme located `src/data`.

## Replicate Experiments From the Paper:

### Dependencies
0. Create a new virtual enviroment
1. Run `pip3 install -r requirements.txt` and `bash nltk_downloader.sh` to load the required packages

### Replicate Results from Table 1:
0. Use the preloaded data files found in `src/data` or re-create them following the `Neural Complexity` steps below.
1. Go into the `src` directory and create the document embedders by running `python3 document_embedder.py`.
2. Run `bash run_n_seeds` to generate the results in `src/results`.
3. Compile the results by running `python3 utils.py --gather_results` to collect the results into `sr/results/final_results.csv`

### Recreate Neural Complexity Model:
#### Part 1: Generate Suprisals Only
0. Split the article text by going into `src` and running `python3 utils.py --split_text`
1. Clone the repo at https://github.com/vansky/neural-complexity.git and follow the instructions to train on Wikitext-2: `time python main.py --model_file 'wiki_2_model.pt' --vocab_file 'wiki_2_vocab.txt' --tied --cuda --data_dir './data/wikitext-2/' --trainfname 'train.txt' --validfname 'valid.txt'`
2. Evaluate on the text from this repo by running inside of the `neural_complexity` folder:
```bash
# gets the non-tokenized version for the LMM over the surprisal
for ((i=1; i<33; i++));
do
time python main.py --model_file 'wiki_2_model.pt' --vocab_file 'wiki_2_vocab.txt' --cuda --data_dir "../src/data/article_texts/" --testfname "$i.txt" --test --words --nopp > "../src/data/article_texts/$i.output"
done

# gets the tokenized version for the sum of the surprisal
for ((i=1; i<33; i++));
do
time python main.py --model_file 'wiki_2_model.pt' --vocab_file 'wiki_2_vocab.txt' --cuda --data_dir "../src/data/article_texts/" --testfname "$i-tok.txt" --test --words --nopp > "../src/data/article_texts/$i.output-tok"
done
```
3. Re-gather the suprisals for each article by running `python3 utils.py --gather_suprisal` inside of `src` which will place the data into `src/data/article_num_to_surprisal_only.csv`

#### Part 2: Train an LMM on the data
0. Go into `src/naturalstories` and run `bash gather_data_from_natural_stories` and `python3 process_natural_stories.py --create_story_files`.  This creates the individual story files.
1. Go to the cloned `neural-complexity` repo and run the script found in `src/naturalstories/model_all_stories.sh` in the root of that cloned repo. 
2. Gather and process the models output by going back to `src/naturalstories` and runnning `python3 process_natural_stories.py --create_data_file_for_lmm --train_lmm`.  This creates the trained LMM model on the Natural Stories corpus and then generates predictions for the new data.

## Citation
If you find this work useful or related to yours, please cite the following:
```
@inproceedings{weller2020you,
  title={You Don’t Have Time to Read This: An Exploration of Document Reading Time Prediction},
  author={Weller, Orion and Hildebrandt, Jordan and Reznik, Ilya and Challis, Christopher and Tass, E Shannon and Snell, Quinn and Seppi, Kevin},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={1789--1794},
  year={2020}
}
```

