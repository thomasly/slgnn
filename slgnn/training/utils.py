import torch
import matplotlib.pyplot as plt


def plot_reconstruct(decoder, encoder, features, adj, labels, index, output):
    fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
    encoder.eval()
    decoder.eval()
    reconst = torch.sigmoid(decoder(
        encoder(
            features[index].cuda(), adj[index].cuda()))).detach().cpu().numpy()

    ax1, ax2 = axes.flatten()
    ax1.bar(list(range(reconst.shape[1])), reconst[0, :])
    ax1.set_xlabel("Reconstructed Fingerprint")
    ax2.bar(list(range(reconst.shape[1])), labels[index])
    ax2.set_xlabel("PubChem Fingerprint")
    fig.savefig(output, dpi=300, bbox_inches="tight")


def plot_train_val_losses(train_losses: list, val_losses: list, output):
    assert(len(train_losses) == len(val_losses))
    fig, axe = plt.subplots(figsize=(8., 6.))
    x = list(range(len(train_losses)))
    axe.plot(x, train_losses, label="train_loss")
    axe.plot(x, val_losses, label="val_loss")
    axe.set_ylabel("BCE loss")
    axe.set_xlabel("Steps")
    axe.legend()
    fig.savefig(output, dpi=300, bbox_inches="tight")
