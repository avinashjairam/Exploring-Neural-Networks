###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

#We are writing our own data loading code. Hence, we are not importing data, unlike, the code we referenced.
import os
import torch
import dill as pickle
import args_utils
import warnings

warnings.filterwarnings('ignore')

#We are changing the arguments and thus defining our own way of utlizing the arguments
# get args
args = args_utils.get_args_generate()

# Set the random seed manually for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

#The code referenced restricted training and generating to either only the CPU or the GPU. We added the following f
#code to allow us to run the training and generation on any device.

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

if os.path.exists(os.path.join(args.data, 'corpus.pkl')):
    with open(os.path.join(args.data, 'corpus.pkl'), 'rb') as f:
        corpus = pickle.load(f)
else:
    raise FileNotFoundError(f'{os.path.join(args.data, "corpus.pkl")} does not exist')

ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

generated = ''
with torch.no_grad():  # no tracking history
    for i in range(args.words):
        if is_transformer_model:
            output = model(input, False)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)
        else:
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

        word = corpus.dictionary.idx2word[word_idx]

        generated = generated + (word + ' ' if word != '<eos>' else '\n')
#The reference code doesn't do any post processing. However, we did so that we do not produce any
#end of sentence tokens which are meaningless in the context of language. 
print(generated)
