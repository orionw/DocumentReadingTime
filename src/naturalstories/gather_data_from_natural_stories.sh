#! /bin/bash
# gathers the data needed to train a LMM on the reading time data from natural stories
wget https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/all_stories.tok
wget https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv