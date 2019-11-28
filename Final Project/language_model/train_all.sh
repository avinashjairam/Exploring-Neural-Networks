#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.2_tied.pt | tee logs/log1.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.2_tied.pt | tee logs/log2.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.2_tied.pt | tee logs/log3.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.3_tied.pt | tee logs/log4.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.3_tied.pt | tee logs/log5.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.3_tied.pt | tee logs/log6.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.4_tied.pt | tee logs/log7.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.4_tied.pt | tee logs/log8.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.4_tied.pt | tee logs/log9.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.5_tied.pt | tee logs/log10.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.5_tied.pt | tee logs/log11.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.5_tied.pt | tee logs/log12.txt
