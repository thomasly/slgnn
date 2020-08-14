from argparse import ArgumentParser


class ModelTrainingArgs(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("-d", "--dataset", help="Dataset name")
        self.add_argument("-c", "--config", help="Config file path")
        self.add_argument(
            "--debug", action="store_true", help="Enable debugging information."
        )
