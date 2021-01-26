from argparse import ArgumentParser
from slgnn.data_processing.zinc_sample_smiles import SmilesSampler


class SamplerArguments(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("-s",
                          "--source",
                          required=True,
                          help="Path to the source file.")
        self.add_argument("-o",
                          "--output",
                          required=True,
                          help="Path to save the output.")
        self.add_argument("-n",
                          "--num-samples",
                          required=True,
                          type=int,
                          help="Number of samples to sample from the source "
                          "file.")
        self.add_argument("-f",
                          "--filter",
                          action="store_true",
                          help="Filter similar SMILES.")
        self.add_argument("-v",
                          "--verbos",
                          action="store_true",
                          help="Show progress bar.")
        self.add_argument("-nh",
                          "--no-header",
                          action="store_false",
                          help="Add this tag if the source file does not have "
                          "header.")
        self.add_argument("-t",
                          "--threshold",
                          type=float,
                          default=0.85,
                          help="Threshold used to filter out similar SMILES.")


if __name__ == "__main__":
    parser = SamplerArguments()
    args = parser.parse_args()
    sampler = SmilesSampler(args.source)
    sampled_smiles = sampler.sample(n_samples=args.num_samples,
                                    filter_similar=args.filter,
                                    threshold=args.threshold,
                                    header=args.no_header,
                                    verbose=args.verbos)
    with open(args.output, "w") as outf:
        outf.write("\n".join(sampled_smiles))
