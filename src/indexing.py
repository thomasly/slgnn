from argparse import ArgumentParser

from slgnn.data_processing import zinc_to_hdf5


parser = ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()
zinc_to_hdf5.ZincToHdf5.indexing(args.path)
