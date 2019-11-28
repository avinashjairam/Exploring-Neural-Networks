import os
import sys
import time
import math
import torch
import torch.nn as nn
import dill as pickle
import data_utils
import model_utils
import args_utils
from tqdm import tqdm

# get args
args = args_utils.get_args_train()

# print args
tqdm.write('{')
for k, v in args.__dict__.items():
    tqdm.write(f'\t{k}: {v}')
tqdm.write('}')
sys.stdout.flush()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

print('loading data..')
if os.path.exists(os.path.join(args.data, 'corpus.pkl')):
    with open(os.path.join(args.data, 'corpus.pkl'), 'rb') as f:
        corpus = pickle.load(f)
else:
    corpus = data_utils.Corpus(args.data)
    with open(os.path.join(args.data, 'corpus.pkl'), 'wb') as f:
        pickle.dump(corpus, f)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    raise NotImplemented('Transformer model not supported')
else:
    model = model_utils.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
        device)

# save the model to disk
with open(args.save, 'wb') as f:
    torch.save(model, f)

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        tqdm_meter = tqdm(range(0, data_source.size(0) - 1, args.bptt),
                          desc='val', unit=' batches', leave=False)
        for i in tqdm_meter:
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            loss = criterion(output_flat, targets)
            total_loss += len(data) * loss.item()
            # update tqdm meter
            tqdm_meter.set_postfix(loss=f'{loss.item():0.4f}')
            tqdm_meter.update()

    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    tqdm_meter = tqdm(range(0, train_data.size(0) - 1, args.bptt),
                      desc=f'[EPOCH {epoch}/{args.epochs}]', unit=' batches', leave=False)
    for batch, i in enumerate(tqdm_meter):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        # update tqdm meter
        tqdm_meter.set_postfix(loss=f'{loss.item():0.4f}', ppl=f'{math.exp(loss.item()):0.2f}')
        tqdm_meter.update()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('starting training..')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        tqdm.write('-' * 89)
        tqdm.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                   'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                              val_loss, math.exp(val_loss)))
        tqdm.write('-' * 89)
        sys.stdout.flush()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    tqdm.write('-' * 89)
    tqdm.write('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
tqdm.write('=' * 89)
tqdm.write('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
tqdm.write('=' * 89)
