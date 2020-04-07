import time
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn.functional as F
import torch.optim as optim

from slgnn.models.gcn.utils import load_encoder_data, load_classifier_data
from slgnn.models.gcn.utils import accuracy
from slgnn.models.gcn.model import GCN
from slgnn.models.decoder.model import Decoder


class TrainingSettings(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument("--no-cuda", action="store_true",
                          help="Disable CUDA.")
        self.add_argument("--fastmode", action="store_true",
                          help="Validate during training pass.")
        self.add_argument("--seed", type=int, default=1733,
                          help="Random seed.")
        self.add_argument("--encoder-epochs", type=int, default=200,
                          help="Number of epochs to train the encoder.")
        self.add_argument("--classifier_epochs", type=int, default=200,
                          help="Number of epochs to train the classifier.")
        self.add_argument("--lr", type=float, default=0.01,
                          help="Initial learning rate.")
        self.add_argument("--weight_decay", type=float, default=5e-4,
                          help="Weight decay (L2 loss on parameters).")
        self.add_argument("--gcn-hidden", type=int, default=20,
                          help="Number of hidden units of the gcn model.")
        self.add_argument("--decoder-hidden", type=int, default=440,
                          help="Number of hidden units of the decoder.")
        self.add_argument("--dropout", type=float, default=0.5,
                          help="Dropout rate (1 - keep probability).")


parser = TrainingSettings()
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def kl_div(p, q):
    return (p * (p / q).log()).sum()


def train_autoencoder(epoch):
    t = time.time()
    decoder.train()
    gcn_model.train()
    encoder_optimizer.zero_grad()
    output = decoder(
        gcn_model(train_encoder["features"], train_encoder["adj"]))
    loss_train = kl_div(output, train_encoder["labels"])
    loss_train.backward()
    encoder_optimizer.step()

    gcn_model.eval()
    decoder.eval()
    output = decoder(gcn_model(val_encoder["features"], val_encoder["adj"]))

    loss_val = F.kl_div(output, val_encoder["labels"])

    print("Epoch: {:>4d}".format(epoch + 1),
          "loss_train: {:.4f}".format(loss_train.item()),
          "loss_val: {:.4f}".format(loss_val.item()),
          "time: {:.4f}s".format(time.time() - t))


def train_classifier(epoch):
    t = time.time()
    classifier.train()
    gcn_model.train()
    finetune_optimizer.zero_grad()
    output = classifier(gcn_model(train_clfr["features"], train_clfr["adj"]))
    loss_train = F.nll_loss(output, train_clfr["labels"])
    acc_train = accuracy(output, train_clfr["labels"])
    loss_train.backward()
    finetune_optimizer.step()

    gcn_model.eval()
    classifier.eval()
    output = classifier(gcn_model(val_clfr["features"], val_clfr["adj"]))
    loss_val = F.nll_loss(output, val_clfr["labels"])
    acc_val = accuracy(output, val_clfr["labels"])
    print("Epoch: {:>4d}".format(epoch + 1),
          "loss_train: {:.4f}".format(loss_train.item()),
          "acc_train: {:.4f}".format(acc_train.item()),
          "loss_val: {:.4f}".format(loss_val.item()),
          "acc_val: {:.4f}".format(acc_val.item()),
          "time: {:.4f}s".format(time.time() - t))


def test():
    gcn_model.eval()
    output = gcn_model(test_clfr["features"], test_clfr["adj"])
    loss_test = F.nll_loss(output, test_clfr["labels"])
    acc_test = accuracy(output, test_clfr["labels"])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train encoder model
# Load data
# the datasets are dictionaries with ["adj", "features", "labels"] keys
train_encoder, val_encoder = load_encoder_data()
# Model and optimizer
gcn_model = GCN(nfeat=train_encoder["features"].shape[1],
                nhid=args.gcn_hidden,
                nclass=args.gcn_hidden,
                dropout=args.dropout)
decoder = Decoder(n_feat=args.gcn_hidden,
                  n_hid=args.decoder_hidden,
                  n_out=881,  # dimension of pubchem fp
                  dropout=args.dropout)
encoder_optimizer = optim.Adam(
    chain(decoder.parameters(), gcn_model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)

if args.cuda:
    gcn_model.cuda()
    decoder.cuda()
    train_encoder = train_encoder.cuda()
    val_encoder = val_encoder.cuda()

t_total = time.time()
for epoch in range(args.encoder_epochs):
    train_autoencoder(epoch)
print("Encoder training finished!")
print("Time elapsed: {:.4f}s".format(time.time() - t_total))

# fine-tune classifier
train_clfr, val_clfr, test_clfr, n_classes = load_classifier_data()
classifier = Decoder(n_feat=args.gcn_hidden,
                     n_hid=args.decoder_hidden,
                     n_out=n_classes,  # dimension of pubchem fp
                     dropout=args.dropout)
finetune_optimizer = optim.Adam(
    chain(classifier.parameters(), gcn_model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)

del train_encoder
del val_encoder
del decoder

if args.cuda:
    torch.cuda.empty_cache()
    gcn_model.cuda()
    classifier.cuda()
    train_clfr = train_clfr.cuda()
    val_clfr = val_clfr.cuda()
    test_clfr = test_clfr.cuda()

for epoch in range(args.classifier_epochs):
    train_classifier(epoch)
print("Fine-tune finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# testing
test()
