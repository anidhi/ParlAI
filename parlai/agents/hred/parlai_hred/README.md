# HRED Agent

Implementation of HRED. 

Model was trained with:

python examples/train_model.py -t dailydialog -m internal:hred -mf [Wherever you want this to be] -nl 2 -rnn gru -hiddensize 512 -esz 256

To run, clone this repo into a folder so that the path to hred.py is at ParlAI/parlai_internal/agents/hred/hred.py

