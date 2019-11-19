import codecs
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from data_utils import MyDataset
from model_utils import MyConvNet
from training_utils import (
    load_checkpoint,
    validate,
    print_model_details
)

if __name__ == "__main__":

    num_classes = 5
    batch_size = 64
    data_dir = "data/gtsrb-german-traffic-sign/Train"
    ckpt_path = "models/adam_optimizer/best.pt"

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data file paths
    filepaths = {}
    for category in ['test']:
        with codecs.open(os.path.join(data_dir, f'{category}.txt'),
                         'r', encoding='utf-8') as f:
            filepaths[category] = list(map(str.strip, f.readlines()))

    tqdm.write('number of samples:')
    for category in ['test']:
        tqdm.write(f'{category}: {len(filepaths[category])}')

    # datasets
    datasets = {}
    for category in ['test']:
        if os.path.exists(f'{data_dir}/{category}.pkl'):
            # load dataset from pickle file
            tqdm.write(f'loading {category} dataset from pickle file..')
            with open(f'{data_dir}/{category}.pkl', 'rb') as f:
                datasets[category] = pickle.load(f)
        else:
            datasets[category] = MyDataset(filepaths=filepaths[category])
            # dump
            tqdm.write(f'writing {category} dataset to pickle file..')
            with open(f'{data_dir}/{category}.pkl', 'wb') as f:
                pickle.dump(datasets[category], f)

    # data loader
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=True)

    # model
    model = MyConvNet(num_classes=num_classes)

    # transfer model to device
    model = model.to(device)

    # print model details
    print_model_details(model)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # load checkpoint
    state = load_checkpoint(ckpt_path, model, device=device)

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
        f'was obtained after {state["best epoch so far"]} epochs'
    )

    # predict
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    # print results
    tqdm.write(f'test loss: {test_loss:0.4f}, test acc: {test_acc:0.2f} %')
