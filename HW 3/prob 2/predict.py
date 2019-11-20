import codecs
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill as pickle
import argparse

from data_utils import TrafficSignsDataset
from model_utils import TrafficSignsConvNet
from training_utils import (
    load_checkpoint,
    validate,
    print_model_details
)

if __name__ == "__main__":

    # get args
    parser = argparse.ArgumentParser(description='train convnet')
    parser.add_argument('--data-dir', type=str, default="data/gtsrb-german-traffic-sign/Train")
    parser.add_argument('--ckpt-path', type=str, default="models/testing/best.pt")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--use-batch-norm', action='store_true', default=False)

    # parse args
    args = parser.parse_args()

    # print args
    tqdm.write('{')
    for k, v in args.__dict__.items():
        tqdm.write('\t{}: {}'.format(k, v))
    tqdm.write('}')

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f'device: {device}')

    # data file paths
    filepaths = {}
    for category in ['test']:
        with codecs.open(os.path.join(args.data_dir, f'{category}.txt'),
                         'r', encoding='utf-8') as f:
            filepaths[category] = list(map(str.strip, f.readlines()))

    tqdm.write('number of samples:')
    for category in ['test']:
        tqdm.write(f'{category}: {len(filepaths[category])}')

    # datasets
    datasets = {}
    for category in ['test']:
        if os.path.exists(f'{args.data_dir}/{category}.pkl'):
            # load dataset from pickle file
            tqdm.write(f'loading {category} dataset from pickle file..')
            with open(f'{args.data_dir}/{category}.pkl', 'rb') as f:
                datasets[category] = pickle.load(f)
        else:
            datasets[category] = TrafficSignsDataset(filepaths=filepaths[category])
            # dump
            tqdm.write(f'writing {category} dataset to pickle file..')
            with open(f'{args.data_dir}/{category}.pkl', 'wb') as f:
                pickle.dump(datasets[category], f)

    # data loader
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=True)

    # model
    model = TrafficSignsConvNet(
        num_classes=args.num_classes,
        batch_norm=args.use_batch_norm
    )

    # transfer model to device
    model = model.to(device)

    # print model details
    print_model_details(model)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # load checkpoint
    tqdm.write(f'loading checkpoint {args.ckpt_path}')
    state = load_checkpoint(args.ckpt_path, model, device=device)

    tqdm.write(
        f'some info on the loaded checkpoint:\n'
        f'(1) it was created after {state["epoch"]} epochs\n'
        f'(2) train loss using this checkpoint is {state["train loss"]:0.4f}\n'
        f'(3) val loss using this checkpoint is {state["val loss"]:0.4f}\n'
        f'(4) train accuracy using this checkpoint is {state["train acc"]:0.2f} %\n'
        f'(5) val accuracy using this checkpoint is {state["val acc"]:0.2f} %\n'
        f'(6) best val accuracy so far '
        f' is {state["best val acc so far"]:0.2f}%\n'
        f'(7) the best val accuracy so far '
        f'was obtained after {state["best epoch so far"]} epochs\n'
    )

    # predict
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    # print results
    tqdm.write(f'test loss: {test_loss:0.4f}, test acc: {test_acc:0.2f} %')
