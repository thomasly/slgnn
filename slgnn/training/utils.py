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
