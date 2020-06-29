## Explanation for the files contained in here:

`articles_and_factor_raw_values.csv` has the articles and their derived features (freq, word_length, etc.) and name.

`screening_study_with_article_info.csv` has all of the derived features, as well as the article text. 

`text_data.csv` only contains the article text and article number.  This is used for the text-only models.

`article_num_to_surprisal_only.csv` contains the article number mapped to the sum of the suprisal of the neural language model

`article_num_to_predictions.json` contains a mapping of the LMM trained on the output from the neural language model
