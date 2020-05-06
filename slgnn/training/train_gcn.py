import os
import time
import argparse
from random import shuffle
from itertools import chain
import pickle as pk

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from slgnn.models.gcn.utils import load_encoder_data, load_classifier_data
from slgnn.models.gcn.utils import accuracy
from slgnn.training.utils import plot_reconstruct
from slgnn.models.gcn.model import GCN
from slgnn.models.decoder.model import Decoder


class TrainingSettings(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--no-cuda",
                          action="store_true",
                          help="Disable CUDA.")
        self.add_argument("--no-encoder",
                          action="store_true",
                          help="Train the classifier without encoder.")
        self.add_argument("--seed",
                          type=int,
                          default=1733,
                          help="Random seed.")
        self.add_argument("--encoder-data",
                          default="data/ZINC/zinc_ghose_1000.hdf5",
                          help="Data path for encoder training.")
        self.add_argument("--encoder-epochs",
                          type=int,
                          default=200,
                          help="Number of epochs to train the encoder.")
        self.add_argument("--classifier-epochs",
                          type=int,
                          default=200,
                          help="Number of epochs to train the classifier.")
        self.add_argument("--batch-size",
                          type=int,
                          default=32,
                          help="Batch size.")
        self.add_argument("--encoder-lr",
                          type=float,
                          default=0.0001,
                          help="Initial learning rate for autoencoder"
                          " training.")
        self.add_argument("--classifier-lr",
                          type=float,
                          default=0.01,
                          help="Initial learning rate for classifier"
                          " training.")
        self.add_argument("--weight-decay",
                          type=float,
                          default=5e-4,
                          help="Weight decay (L2 loss on parameters).")
        self.add_argument("--gcn-hidden",
                          type=int,
                          default=100,
                          help="Number of hidden units of the gcn model.")
        self.add_argument("--decoder-hidden",
                          type=int,
                          default=440,
                          help="Number of hidden units of the decoder.")
        self.add_argument("--dropout",
                          type=float,
                          default=0.1,
                          help="Dropout rate (1 - keep probability).")


parser = TrainingSettings()
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def train_autoencoder(epoch, batch_size=32):
    global encoder_optimizer
    if (epoch + 1) % 2 == 0:
        for param_group in encoder_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, 1e-10)
    t = time.time()
    decoder.train()
    gcn_model.train()
    steps = 0
    train_losses = []
    loss_train = torch.Tensor([0]).cuda()
    data = list(
        zip(train_encoder["features"], train_encoder["adj"],
            train_encoder["labels"]))
    shuffle(data)
    for feat, adj, lb in data:
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        output = decoder(gcn_model(feat, adj))
        # print(f"output: {output}")
        # print(f"label: {lb}")
        # one_step_loss = torch.abs(torch.sigmoid(output) - lb).sum()
        # one_step_loss = torch.sqrt(torch.pow((output - lb), 2)).sum()
        one_step_loss = F.binary_cross_entropy_with_logits(output, lb)
        loss_train += one_step_loss
        steps += 1
        if steps % batch_size == 0:
            loss_train = loss_train / batch_size
            encoder_optimizer.zero_grad()
            loss_train.backward()
            encoder_optimizer.step()
            train_losses.append(loss_train.item())
            print("Step: {:>4}".format(steps),
                  "loss_train: {:.4f}".format(loss_train.item()),
                  end="\r")
            loss_train = torch.Tensor([0]).cuda()

    gcn_model.eval()
    decoder.eval()
    steps = 0
    loss_val = 0.
    for feat, adj, lb in zip(val_encoder["features"], val_encoder["adj"],
                             val_encoder["labels"]):
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        output = decoder(gcn_model(feat, adj))

        # loss = torch.abs(torch.sigmoid(output) - lb).sum()
        loss = F.binary_cross_entropy_with_logits(output, lb)
        loss_val += loss.item()
        steps += 1
    loss_val = loss_val / steps
    print(
        "Epoch: {:>4d}".format(epoch + 1),
        #   "loss_train: {:.4f}".format(loss_train.item()),
        "loss_val: {:.4f}".format(loss_val),
        "time: {:.4f}s".format(time.time() - t))

    return train_losses, loss_val


def train_classifier(epoch, batch_size=32):
    global finetune_optimizer
    if (epoch + 1) % 2 == 0:
        for param_group in finetune_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, 1e-10)
    t = time.time()
    classifier.train()
    gcn_model.train()
    loss_train = torch.Tensor([0]).cuda()
    acc_train = 0.
    steps = 0
    data = list(
        zip(train_clfr["features"], train_clfr["adj"], train_clfr["labels"]))
    shuffle(data)
    for feat, adj, lb in data:
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        if epoch < 2:
            with torch.no_grad():
                output = gcn_model(feat, adj)
        else:
            output = gcn_model(feat, adj)
        output = classifier(output)
        loss_train_onestep = F.binary_cross_entropy_with_logits(output, lb)
        loss_train += loss_train_onestep
        acc_train += accuracy(output, lb)
        steps += 1
        if steps % batch_size == 0:
            loss_train = loss_train / batch_size
            finetune_optimizer.zero_grad()
            loss_train.backward()
            finetune_optimizer.step()
            print(
                "Steps: {:>5d}".format(steps),
                "loss_train: {:.4f}".format(loss_train.item()),
                #   "acc_train: {} / {}".format(acc_train, batch_size),
                "acc_train: {:.4f}".format(acc_train / batch_size),
                end="\r")
            loss_train = torch.Tensor([0]).cuda()
            acc_train = 0.

    gcn_model.eval()
    classifier.eval()
    loss_val, acc_val = 0., 0.
    steps = 0
    for feat, adj, lb in zip(val_clfr["features"], val_clfr["adj"],
                             val_clfr["labels"]):
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        output = classifier(gcn_model(feat, adj))
        loss_val_one_step = F.binary_cross_entropy_with_logits(output, lb)
        acc_val_one_step = accuracy(torch.sigmoid(output), lb)
        loss_val += loss_val_one_step
        acc_val += acc_val_one_step
        steps += 1

    print(
        "Epoch: {:>4d}".format(epoch + 1),
        "loss_val: {:.4f}".format(loss_val / steps),
        #   "acc_val: {}/{}".format(acc_val, steps),
        "acc_val: {:.4f}".format(acc_val / steps),
        "time: {:.4f}s".format(time.time() - t),
        end="\n")


def test():
    gcn_model.eval()
    classifier.eval()
    loss_test, acc_test = 0., 0.
    steps = 0
    for feat, adj, lb in zip(test_clfr["features"], test_clfr["adj"],
                             test_clfr["labels"]):
        feat = feat.cuda()
        adj = adj.cuda()
        lb = lb[None, :].cuda()
        output = classifier(gcn_model(feat, adj))
        loss_test += F.binary_cross_entropy_with_logits(output, lb).item()
        acc_test += accuracy(torch.sigmoid(output), lb)
        steps += 1

    print(
        "Testing results:",
        "loss= {:.4f}".format(loss_test / steps),
        #   "accuracy= {} / {}".format(acc_test, steps))
        "accuracy= {:.4f}".format(acc_test / steps),
        end="\r")


# Train encoder model
# Load data
# the datasets are dictionaries with ["adj", "features", "labels"] keys
if not args.no_encoder:
    train_encoder, val_encoder, len_fp = load_encoder_data(args.encoder_data,
                                                           type_="txt")
    # Model and optimizer
    gcn_model = GCN(nfeat=train_encoder["features"].shape[2],
                    nhid=args.gcn_hidden,
                    nclass=args.gcn_hidden,
                    dropout=args.dropout)
    decoder = Decoder(
        n_feat=args.gcn_hidden,
        n_hid=args.decoder_hidden,
        n_out=len_fp,  # dimension of pubchem fp
        dropout=args.dropout)
    encoder_optimizer = optim.Adam(chain(decoder.parameters(),
                                         gcn_model.parameters()),
                                   lr=args.encoder_lr,
                                   weight_decay=args.weight_decay)

    if args.cuda:
        gcn_model.cuda()
        decoder.cuda()

    t_total = time.time()
    losses_train, losses_val = list(), list()
    for epoch in range(args.encoder_epochs):
        l_train, l_val = train_autoencoder(epoch, args.batch_size)
        losses_train.extend(l_train)
        losses_val.append(l_val)

    fig, axes = plt.subplots(1, 2, figsize=(16., 6.))
    axes[0].plot(list(range(len(losses_train))), losses_train)
    axes[0].set(xlabel="Steps", ylabel="Training loss")
    axes[1].plot(list(range(len(losses_val))), losses_val, color="orange")
    axes[1].set(xlabel="Epochs",
                ylabel="Validation loss",
                xlim=[0, args.encoder_epochs])
    os.makedirs("logs", exist_ok=True)
    fig.savefig("logs/autoencoder_training.png", dpi=300, bbox_inches="tight")
    with open("logs/autoencoder_training_losses.pk", "wb") as f:
        pk.dump({"train_loss": losses_train, "val_loss": losses_val}, f)
    os.makedirs("trained_models", exist_ok=True)
    torch.save(gcn_model, "trained_models/gcn_model.pt")
    torch.save(
        {
            "gcn_model_state_dict": gcn_model.state_dict(),
            "decoder_model_state_dict": decoder.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": encoder_optimizer.state_dict()
        }, "trained_models/gcn_model_checkpoint")
    print("Encoder training finished!")
    plot_reconstruct(decoder, gcn_model, train_encoder["features"],
                     train_encoder["adj"], train_encoder["labels"], 0,
                     "logs/autoencoder_reconstruct_0.png")
    plot_reconstruct(decoder, gcn_model, train_encoder["features"],
                     train_encoder["adj"], train_encoder["labels"], 10,
                     "logs/autoencoder_reconstruct_10.png")
    print("Time elapsed: {:.4f}s".format(time.time() - t_total))

# fine-tune classifier
train_clfr, val_clfr, test_clfr, n_classes = load_classifier_data(
    "data/sider.csv.gz",
    label_cols=[
        'Hepatobiliary disorders',
        # 'Metabolism and nutrition disorders',
        # 'Product issues', 'Eye disorders', 'Investigations',
        # 'Musculoskeletal and connective tissue disorders',
        # 'Gastrointestinal disorders', 'Social circumstances',
        # 'Immune system disorders',
        # 'Reproductive system and breast disorders',
        # 'Neoplasms benign,
        # malignant and unspecified (incl cysts and polyps)',
        # 'General disorders and administration site conditions',
        # 'Endocrine disorders', 'Surgical and medical procedures',
        # 'Vascular disorders', 'Blood and lymphatic system disorders',
        # 'Skin and subcutaneous tissue disorders',
        # 'Congenital, familial and genetic disorders',
        # 'Infections and infestations',
        # 'Respiratory, thoracic and mediastinal disorders',
        # 'Psychiatric disorders', 'Renal and urinary disorders',
        # 'Pregnancy, puerperium and perinatal conditions',
        # 'Ear and labyrinth disorders', 'Cardiac disorders',
        # 'Nervous system disorders',
        # 'Injury, poisoning and procedural complications'
    ])
if args.no_encoder:
    gcn_model = GCN(nfeat=train_clfr["features"].shape[2],
                    nhid=args.gcn_hidden,
                    nclass=args.gcn_hidden,
                    dropout=args.dropout)
    t_total = time.time()
classifier = Decoder(n_feat=args.gcn_hidden,
                     n_hid=args.decoder_hidden,
                     n_out=n_classes,
                     dropout=args.dropout)
finetune_optimizer = optim.Adam(chain(classifier.parameters(),
                                      gcn_model.parameters()),
                                lr=args.classifier_lr,
                                weight_decay=args.weight_decay)

if args.cuda:
    torch.cuda.empty_cache()
    gcn_model.cuda()
    classifier.cuda()

for epoch in range(args.classifier_epochs):
    train_classifier(epoch, args.batch_size)
print("Fine-tune finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# testing
test()
