from argparse import ArgumentParser

from slgnn.data_processing.zinc_to_hdf5 import ZincToHdf5


parser = ArgumentParser()
parser.add_argument("-p", "--path", help="Path to the dataset folder.")
parser.add_argument("-n", "--n-samples", type=int,
                    help="Number of samples to sample from the dataset.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Show progress bar.")
parser.add_argument("-o", "--output", help="Output file path.")
args = parser.parse_args()

saver = ZincToHdf5.random_sample_without_index(
    args.n_samples, args.path, verbose=True)
saver.save_hdf5(args.output)
