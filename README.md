Homework 4 - Deep Learning Methods for Automated Discourse

Train a model on OpenSubtitles using P(T|S) 

python -m parlai.scripts.multiprocessing_train -t opensubtitles -mf parlai_internal/forward_model.ckpt -bs 16 -m transformer/generator -stim 7200 -sval True -emb glove --beam-size 40 --inference beam -opt adam -esz 300 -nl 4 -dr 0.3 -bi True -att local --attention-time pre --rnn-class lstm --decoder same -bs 16 -lr 0.001 --max-train-time 10800 --save-every-n-secs  300 -mcs all -df OpenSubtitlesVersion2018/opensubtitles.dict 

Train a model on OpenSubtitles using P(S|T)

python -m parlai.scripts.multiprocessing_train -t opensubtitles -mf parlai_internal/backward_model.ckpt -bs 16 -m transformer/generator -stim 7200 -sval True -emb glove --beam-size 40 --inference beam -opt adam -esz 300 -nl 4 -dr 0.3 -bi True -att local --attention-time pre --rnn-class lstm --decoder same -bs 16 -lr 0.001 --max-train-time 10800 --save-every-n-secs  300 -mcs all -df OpenSubtitlesVersion2018/opensubtitles.dict 

Train P(T|S) transformer model on the DialyDialog dataset

python -m parlai.scripts.multiprocessing_train -t dailydialog -mf parlai_internal/forward_modelTunedOnDialog_model.ckpt -bs 16 -m transformer/generator -stim 7200 -sval True -emb glove --beam-size 40 --inference beam -opt adam -esz 300 -nl 4 -dr 0.3 -bi True -att local --attention-time pre --rnn-class lstm --decoder same -bs 16 -lr 0.001 --max-train-time 10800 --save-every-n-secs  300 -mcs all -df OpenSubtitlesVersion2018/opensubtitles.dict 


Implement decoding with MMI-bidi objective


Evaluate your chatbot on the mechanical turk SANDBOX using ParlAI’s model evaluator

