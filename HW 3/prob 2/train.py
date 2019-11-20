import torch
from torch import nn, optim
from tqdm import tqdm
import codecs
import os
import dill as pickle
import argparse

from torch.utils.data import DataLoader
from model_utils import TrafficSignsConvNet
from data_utils import TrafficSignsDataset
from training_utils import (
    save_checkpoint,
    train,
    validate,
    print_model_details
)

if __name__ == "__main__":

    # get args
    parser = argparse.ArgumentParser(description='train convnet')
    parser.add_argument('--data-dir', type=str, default="data/gtsrb-german-traffic-sign/Train")
    parser.add_argument('--ckpt-dir', type=str, default="models/testing")
    parser.add_argument('--optimizer-name', type=str, default='adam',
                        help='options: {"adam", "SGD_with_nesterov"}')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--dropout-prob', type=float, default=0.3)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--use-xavier-init', action='store_true', default=False)
    parser.add_argument('--use-batch-norm', action='store_true', default=False)
    parser.add_argument('--use-lr-scheduler', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')

    # parse args
    args = parser.parse_args()

    # print args
    tqdm.write('{')
    for k, v in args.__dict__.items():
        tqdm.write('\t{}: {}'.format(k, v))
    tqdm.write('}')

    # create logger
    logger = {
        'train losses': [],
        'val losses': [],
        'train accs': [],
        'val accs': []
    }

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f'device: {device}')

    # set random seed
    torch.manual_seed(args.seed) if device == torch.device('cuda:0') \
        else torch.cuda.manual_seed_all(args.seed)

    # data file paths
    filepaths = {}
    for category in ['train', 'val']:
        with codecs.open(os.path.join(args.data_dir, f'{category}.txt'),
                         'r', encoding='utf-8') as f:
            filepaths[category] = list(map(str.strip, f.readlines()))

    tqdm.write('number of samples:')
    for category in ['train', 'val']:
        tqdm.write(f'{category}: {len(filepaths[category])}')

    # datasets
    datasets = {}
    for category in ['train', 'val']:
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

    # data loaders
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=True)

    # model
    model = TrafficSignsConvNet(
        num_classes=args.num_classes,
        batch_norm=args.use_batch_norm,
        dropout=args.dropout_prob,
        xavier=args.use_xavier_init
    )

    # transfer model to device
    model = model.to(device)

    # print model details
    print_model_details(model)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.optimizer_name == 'adam':
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
    elif args.optimizer_name == 'SGD_with_nesterov':
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise NotImplemented(f'{args.optimizer_name}')

    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=0, verbose=True  # mode 'max' to maximize val acc
        )

    best_val_acc = -1
    best_epoch = -1
    # train for epochs
    tqdm.write('training..')
    for epoch in range(1, args.num_epochs + 1):
        # train
        train_loss, train_acc = train(
            model, train_loader,
            criterion, optimizer,
            epoch, args.num_epochs, device
        )
        # validate
        val_loss, val_acc = validate(
            model, val_loader,
            criterion, device
        )

        # add losses and accs to logger
        logger['train losses'].append(train_loss)
        logger['val losses'].append(val_loss)
        logger['train accs'].append(train_acc)
        logger['val accs'].append(val_acc)

        if val_acc >= best_val_acc:
            tqdm.write('    - found new best validation accuracy\n')
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

        if args.use_lr_scheduler:
            # adjust learning rate
            scheduler.step(val_acc, epoch=epoch)
            state['scheduler_dict'] = scheduler.state_dict()

        # save state
        save_checkpoint(state, is_best, args.ckpt_dir)

    # save logger
    tqdm.write('dumping logger to pickle file..')
    with open(f'{args.ckpt_dir}/logger.pkl', 'wb') as f:
        pickle.dump(logger, f)

    tqdm.write(f'max validation accuracy {best_val_acc:0.2f} % '
               f'was obtained after {best_epoch} epochs')
