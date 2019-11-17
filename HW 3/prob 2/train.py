import torch
from torch import nn, optim
from tqdm import tqdm
import codecs
import os

from torch.utils.data import DataLoader
from model_utils import MyConvNet
from data_utils import MyDataset
from training_utils import (
    save_checkpoint,
    train,
    validate,
    print_model_details
)

if __name__ == "__main__":

    num_classes = 20
    num_epochs = 10
    learning_rate = 1e-3
    batch_size = 64
    data_dir = "data"
    ckpt_dir = "models/adam_optimizer"

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(123) if device == torch.device('cuda:0') \
        else torch.cuda.manual_seed(123)

    # data file paths
    filepaths = {}
    for category in ['train', 'val']:
        with codecs.open(os.path.join(data_dir, f'{category}.txt'),
                         'r', encoding='utf-8') as f:
            filepaths[category] = list(map(str.strip, f.readlines()))

    tqdm.write('number of samples:')
    for category in ['train', 'val']:
        tqdm.write(f'{category}: {len(filepaths[category])}')

    # datasets
    datasets = {}
    for category in ['train', 'val']:
        datasets[category] = MyDataset(filepaths=filepaths[category])

    # data loaders
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=True)

    # model
    model = MyConvNet(num_classes=num_classes)

    # transfer model to device
    model = model.to(device)

    # print model details
    print_model_details(model)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    best_val_acc = -1
    best_epoch = -1
    # train for epochs
    for epoch in range(1, num_epochs + 1):
        # train
        train_loss, train_acc = train(
            model, train_loader,
            criterion, optimizer,
            epoch, num_epochs, device
        )
        # validate
        val_loss, val_acc = validate(
            model, val_loader,
            criterion, device
        )

        if val_acc >= best_val_acc:
            tqdm.write('    - found new best validation accuracy')
            best_val_acc = val_acc
            best_epoch = epoch
            is_best = True
        else:
            is_best = False

        # current state
        state = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val loss': val_loss,
            'val acc': val_acc,
            'train loss': train_loss,
            'train acc': train_acc,
            'best val acc so far': best_val_acc,
            'best epoch so far': best_epoch
        }

        # save state
        save_checkpoint(state, is_best, ckpt_dir)

    tqdm.write(f'max validation accuracy {best_val_acc} % '
               f'was obtained after {best_epoch} epochs')
