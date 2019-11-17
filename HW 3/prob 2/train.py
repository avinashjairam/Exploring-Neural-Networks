import torch
from tqdm import tqdm
from model_utils import MyConvNet
from data_utils import MyDataset
from training_utils import save_checkpoint, load_checkpoint


def train(model, data_loader, criterion, optimizer, curr_epoch, tot_epochs, device=torch.device('cpu')):
    """
    train for an epoch
    :param model: (torch.nn.module) model
    :param data_loader: (torch.data.Dataloader) data loader
    :param optimizer: (torch.optim) optimizer
    :param criterion: (torch.nn.module) criterion
    :param curr_epoch: current epoch
    :param tot_epochs: total number of epochs
    :param device: (torch.device) device (default: torch.device('cpu'))
    :return: training loss, training acc
    """

    # set model to training mode
    model.train()

    # progress meter
    tqdm_meter = tqdm(
        data_loader,
        desc=f'[Epoch {curr_epoch}/{tot_epochs}]',
        unit=' batches'
    )

    corrects = 0
    total = 0
    tot_loss = 0

    for idx, (image, label) in enumerate(tqdm_meter):
        # transfer to device
        image, label = image.to(device), label.to(device)

        # zero out gradients
        optimizer.zero_grad()

        # forward pass
        out = model(image)

        # compute loss
        loss = criterion(out, label)

        # update total loss
        tot_loss = tot_loss + loss.item()

        # update tqdm meter
        tqdm_meter.set_postfix(f'')

        # backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # compute prediction
        pred = torch.multinomial(out, -1)

        # compute number of correct predictions and update corrects
        corrects = corrects + (pred == label).sum().item()

        total = total + label.shape[0]
        # TODO
