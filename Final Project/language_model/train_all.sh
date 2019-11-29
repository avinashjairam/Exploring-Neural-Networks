#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --nhid 100 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.2_tied.pt | tee logs/lstm_emsize100_dropout0.2_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --nhid 200 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.2_tied.pt | tee logs/lstm_emsize200_dropout0.2_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --nhid 300 --dropout 0.2 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.2_tied.pt | tee logs/lstm_emsize300_dropout0.2_tied.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --nhid 100 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.3_tied.pt | tee logs/lstm_emsize100_dropout0.3_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --nhid 200 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.3_tied.pt | tee logs/lstm_emsize200_dropout0.3_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --nhid 300 --dropout 0.3 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.3_tied.pt | tee logs/lstm_emsize300_dropout0.3_tied.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --nhid 100 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.4_tied.pt | tee logs/lstm_emsize100_dropout0.4_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --nhid 200 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.4_tied.pt | tee logs/lstm_emsize200_dropout0.4_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --nhid 300 --dropout 0.4 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.4_tied.pt | tee logs/lstm_emsize300_dropout0.4_tied.txt

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 100 --nhid 100 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize100_dropout0.5_tied.pt | tee logs/lstm_emsize100_dropout0.5_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 200 --nhid 200 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize200_dropout0.5_tied.pt | tee logs/lstm_emsize200_dropout0.5_tied.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --batch_size 256 --emsize 300 --nhid 300 --dropout 0.5 --tied --model LSTM --epochs 40 --save models/lstm_emsize300_dropout0.5_tied.pt | tee logs/lstm_emsize300_dropout0.5_tied.txt
