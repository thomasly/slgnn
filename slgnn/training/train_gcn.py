import time
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
        self.add_argument("--lr", type=float, default=0.00001,
                          help="Initial learning rate.")
        self.add_argument("--weight_decay", type=float, default=5e-4,
                          help="Weight decay (L2 loss on parameters).")
        self.add_argument("--gcn-hidden", type=int, default=20,
                          help="Number of hidden units of the gcn model.")
        self.add_argument("--decoder-hidden", type=int, default=440,
                          help="Number of hidden units of the decoder.")
        self.add_argument("--dropout", type=float, default=0.1,
                          help="Dropout rate (1 - keep probability).")


parser = TrainingSettings()
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def kl_div(p, q):
    return (p * (p / q).log()).sum()


def train_autoencoder(epoch, batch_size=32):
    t = time.time()
    decoder.train()
    gcn_model.train()
    encoder_optimizer.zero_grad()
    steps = 0
    train_losses = []
    loss_train = torch.Tensor([0]).cuda()
    for feat, adj, lb in zip(train_encoder["features"],
                             train_encoder["adj"],
                             train_encoder["labels"]):
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        output = decoder(gcn_model(feat, adj))
        one_step_loss = torch.abs(output - lb).sum()
        loss_train += one_step_loss
        steps += 1
        if steps % batch_size == 0:
            loss_train = loss_train / batch_size
            loss_train.backward()
            encoder_optimizer.step()
            train_losses.append(loss_train.item())
            print("Step: {}, loss_train: {}".format(steps, loss_train.item()),
                  end="\r")
            loss_train = torch.Tensor([0]).cuda()

    gcn_model.eval()
    decoder.eval()
    steps = 0
    loss_val = 0
    for feat, adj, lb in zip(val_encoder["features"],
                             val_encoder["adj"],
                             val_encoder["labels"]):
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb.cuda()
        output = decoder(gcn_model(feat, adj))

        loss = torch.abs(output - lb).sum()
        loss_val += loss.item()
        steps += 1
    loss_val = loss_val / steps
    print("Epoch: {:>4d}".format(epoch + 1),
          #   "loss_train: {:.4f}".format(loss_train.item()),
          "loss_val: {:.4f}".format(loss_val),
          "time: {:.4f}s".format(time.time() - t))

    return train_losses, loss_val


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
train_encoder, val_encoder = load_encoder_data(
    "test_data/zinc_ghose_1000.hdf5")
# Model and optimizer
gcn_model = GCN(nfeat=train_encoder["features"].shape[2],
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

t_total = time.time()
losses_train, losses_val = list(), list()
for epoch in range(args.encoder_epochs):
    l_train, l_val = train_autoencoder(epoch)
    losses_train.extend(l_train)
    losses_val.append(l_val)

fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))
axes[0].plot(list(range(len(losses_train))), losses_train)
axes[0].set(xlabel="Steps", ylabel="Training loss")
axes[1].plot(list(range(len(losses_val))), losses_val, color="orange")
axes[1].set(xlabel="Epochs",
            ylabel="Validation loss",
            xlim=[0, args.encoder_epochs])
fig.savefig("autoencoder_training.png")

print("Encoder training finished!")
print("Time elapsed: {:.4f}s".format(time.time() - t_total))

# fine-tune classifier
train_clfr, val_clfr, test_clfr, n_classes = load_classifier_data(
    "test_data/tox21.csv.gz",
    label_cols=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                'SR-MMP', 'SR-p53'])
classifier = Decoder(n_feat=args.gcn_hidden,
                     n_hid=args.decoder_hidden,
                     n_out=n_classes,  # dimension of pubchem fp
                     dropout=args.dropout)
finetune_optimizer = optim.Adam(
    chain(classifier.parameters(), gcn_model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)

# del train_encoder
# del val_encoder
# del decoder

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
