import torch
import os
import shutil


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
        os.mkdir(checkpoint_dir)
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
    :return: nothing
    """
    if not os.path.exists(checkpoint_path):
        raise ("File doesn't exist {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
