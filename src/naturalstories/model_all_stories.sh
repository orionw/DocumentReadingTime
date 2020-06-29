#! /bin/bash
for ((i=1; i<11; i++));
do
time python main.py --model_file 'wiki_2_model.pt' --vocab_file 'wiki_2_vocab.txt' --cuda --data_dir "../src/naturalstories/ind_stories/" --testfname "$i.txt" --test --words --nopp > ".../src/naturalstories/ind_stories/$i.output"
done