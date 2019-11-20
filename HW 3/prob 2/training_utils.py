import torch
import os
import shutil
from tqdm import tqdm


def train(
        model,
        data_loader,
        criterion,
        optimizer,
        curr_epoch,
        tot_epochs,
        device=torch.device('cpu')
):
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
        desc=f'🏍️[Epoch {curr_epoch}/{tot_epochs}]🏍️',
        unit=' batches',
        leave=False
    )

    corrects = 0
    total = 0
    tot_loss = 0

    for batch_idx, (image, label) in enumerate(tqdm_meter):
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
        tqdm_meter.set_postfix(ordered_dict={"loss": f"{loss.item():0.4f}"})
        tqdm_meter.update()

        # backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # compute prediction
        _, pred = torch.max(out, dim=1)

        # compute number of correct predictions and update corrects
        corrects = corrects + (pred == label).sum().item()

        # update total
        total = total + label.shape[0]

    assert batch_idx + 1 == len(data_loader), 'number of batches mismatch'
    assert total == len(data_loader.dataset), 'number of samples mismatch'

    av_loss = tot_loss / len(data_loader)
    acc = (100 * corrects) / total  # accuracy in percentage

    # print
    tqdm.write(f'[Epoch {curr_epoch}/{tot_epochs}] loss: {av_loss:0.4f}, '
               f'acc: {acc:0.2f} %')
    return av_loss, acc


def validate(
        model,
        data_loader,
        criterion,
        device=torch.device('cpu'),
        predictions_labels_lists=False
):
    """
    validate
    :param model: (torch.nn.module) model
    :param data_loader: (torch.data.Dataloader) data loader
    :param criterion: (torch.nn.module) criterion
    :param device: (torch.device) device (default: torch.device('cpu'))
    :param predictions_labels_lists: (bool) return predictions and labels in lists or not (default: False)
    :return: val loss, val acc (also preds list and labels list if predictions_labels_lists is True)
    """

    # set model to eval mode
    model.eval()

    # progress meter
    tqdm_meter = tqdm(
        data_loader,
        desc=f'val',
        unit=' batches',
        leave=False
    )

    corrects = 0
    total = 0
    tot_loss = 0

    if predictions_labels_lists:
        preds_list, labels_list = [], []

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm_meter):
            # transfer to device
            image, label = image.to(device), label.to(device)

            # forward pass
            out = model(image)

            # compute loss
            loss = criterion(out, label)

            # update total loss
            tot_loss = tot_loss + loss.item()

            # update tqdm meter
            tqdm_meter.set_postfix(ordered_dict={"loss": f"{loss.item():0.4f}"})
            tqdm_meter.update()

            # compute prediction
            _, pred = torch.max(out, dim=1)

            # compute number of correct predictions and update corrects
            corrects = corrects + (pred == label).sum().item()

            if predictions_labels_lists:
                # add to preds list
                preds_list.extend(pred.detach().cpu().tolist())
                labels_list.extend(label.detach().cpu().tolist())

            # update total
            total = total + label.shape[0]

    assert batch_idx + 1 == len(data_loader), 'number of batches mismatch'
    assert total == len(data_loader.dataset), 'number of samples mismatch'

    av_loss = tot_loss / len(data_loader)
    acc = (100 * corrects) / total  # accuracy in percentage

    # print
    tqdm.write(f'\t(val) loss: {av_loss:0.4f}, '
               f'acc: {acc:0.2f} %\n')

    if predictions_labels_lists:
        return av_loss, acc, preds_list, labels_list
    else:
        return av_loss, acc


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    save checkpoint
    :param state: state
    :param is_best: is it the best checkpoint?
    :param checkpoint_dir: path to the checkpoint directory
    :return: nothing
    """
    filepath = os.path.join(checkpoint_dir, 'last.pt')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pt'))


def load_checkpoint(checkpoint_path, model, optimizer=None, device=torch.device('cpu')):
    """
    load checkpoint
    :param checkpoint_path: path to checkpoint
    :param model: (torch.nn.Module) model
    :param optimizer: (torch.optim) optimizer
    :param device: (torch.device) device
    :return: (torch.nn.module) checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise ("File doesn't exist {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def print_model_details(model):
    # model details
    print(89 * '-')
    print('model:')
    print(89 * '-')
    print(model)
    print(89 * '-')
    print('number of trainable params:', sum(p.numel() for p in model.parameters()
                                             if p.requires_grad))
    print(89 * '-')
