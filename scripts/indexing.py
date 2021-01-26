from argparse import ArgumentParser

from slgnn.data_processing import zinc_to_hdf5


parser = ArgumentParser()
parser.add_argument("-p", "--path", help="Path to the directory.")
parser.add_argument("-b", "--progress-bar", action="store_true",
                    help="Show progress bar.")
args = parser.parse_args()
zinc_to_hdf5.ZincToHdf5.indexing(args.path, args.progress_bar)
