from pathlib import Path
import json
import pickle
from copy import deepcopy

from torch.optim import Adam, SGD
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR  # , ReduceLROnPlateau
import yaml

from slgnn.data_processing.pyg_datasets import (
    ZINC1k,
    ZINC10k,
    ZINC100k,
    JAK1,
    JAK2,
    JAK3,
    JAK1FP,
    JAK2FP,
    JAK3FP,
    JAK1Presplitted,
    JAK2Presplitted,
    JAK3Presplitted,
    JAK1Dude,
    JAK2Dude,
    JAK3Dude,
)
from slgnn.data_processing.deepchem_datasets import (
    Sider,
    SiderFP,
    BBBP,
    BBBPFP,
    BACE,
    BACEFP,
    HIV,
    HIVFP,
    HIVBalanced,
    ClinTox,
    ClinToxFP,
    ClinToxBalanced,
    Amu,
    AmuFP,
    Ellinger,
    EllingerFP,
    Mpro,
    MproFP,
    RepurposingFP,
)
from slgnn.training.utils import Patience
from slgnn.metrics.metrics import Accuracy, ROC_AUC, F1, AP
from slgnn.models.gcn.model import GIN, CPAN


def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")


class ConfigError(Exception):
    pass


class Config:
    """
    Specifies the configuration for a single model.
    """

    encoder_datasets = {
        "ZINC1k": ZINC1k,
        "ZINC10k": ZINC10k,
        "ZINC100k": ZINC100k,
        "JAK1": JAK1FP,
        "JAK2": JAK2FP,
        "JAK3": JAK3FP,
        "JAK1Dude": JAK1Dude,
        "JAK2Dude": JAK2Dude,
        "JAK3Dude": JAK3Dude,
        "Sider": SiderFP,
        "BACE": BACEFP,
        "BBBP": BBBPFP,
        "ClinTox": ClinToxFP,
        "HIV": HIVFP,
        "Repurposing": RepurposingFP,
        "Amu": AmuFP,
        "Ellinger": EllingerFP,
        "Mpro": MproFP,
    }

    classifier_datasets = {
        "JAK1": JAK1,
        "JAK2": JAK2,
        "JAK3": JAK3,
        "JAK1Pre": JAK1Presplitted,
        "JAK2Pre": JAK2Presplitted,
        "JAK3Pre": JAK3Presplitted,
        "Sider": Sider,
        "BACE": BACE,
        "BBBP": BBBP,
        "ClinTox": ClinTox,
        "ClinToxBalanced": ClinToxBalanced,
        "HIV": HIV,
        "HIVBalanced": HIVBalanced,
        "Amu": Amu,
        "Ellinger": Ellinger,
        "Mpro": Mpro,
    }

    models = {
        "CPAN": CPAN,
        "GIN": GIN,
        # 'ECC': ECC,
        # "DiffPool": DiffPool,
        # "DGCNN": DGCNN,
        # "MolecularFingerprint": MolecularFingerprint,
        # "DeepMultisets": DeepMultisets,
        # "GraphSAGE": GraphSAGE
    }

    encoder_losses = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        # 'MulticlassClassificationLoss': MulticlassClassificationLoss,
        # 'NN4GMCLoss': NN4GMulticlassClassificationLoss,
        # 'DMCL': DiffPoolMulticlassClassificationLoss,
    }

    classifier_losses = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
    }

    metrics = {
        "Accuracy": Accuracy,
        "ROC_AUC": ROC_AUC,
        "F1": F1,
        "Average_precision": AP,
    }

    optimizers = {
        "Adam": Adam,
        "SGD": SGD,
    }

    schedulers = {
        "StepLR": StepLR,
        # 'ECCLR': ECCLR,
        # 'ReduceLROnPlateau': ReduceLROnPlateau
    }

    early_stoppers = {
        # 'GLStopper': GLStopper,
        "Patience": Patience
    }

    def __init__(self, **attrs):

        # print(attrs)
        self.config = dict(attrs)

        for attrname, value in attrs.items():
            attrnames = [
                "encoder_dataset",
                "classifier_dataset",
                "model",
                "encoder_loss",
                "classifier_loss",
                "optimizer",
                "scheduler",
                "encoder_early_stopper",
                "early_stopper",
                "metrics",
            ]
            if attrname in attrnames:
                if attrname == "encoder_dataset":
                    setattr(self, "encoder_dataset_name", "_".join(value))
                if attrname == "classifier_dataset":
                    setattr(self, "classifier_dataset_name", value)
                if attrname == "model":
                    setattr(self, "model_name", value["class"])
                if attrname == "encoder_loss":
                    setattr(self, "encoder_loss_name", value)
                if attrname == "classifier_loss":
                    setattr(self, "classifier_loss_name", value)
                if attrname == "metrics":
                    setattr(self, "metrics_name", value)

                fn = getattr(self, f"parse_{attrname}")
                setattr(self, attrname, fn(value))
            else:
                setattr(self, attrname, value)

    def __getitem__(self, name):
        # print("attr", name)
        return getattr(self, name)

    def __contains__(self, attrname):
        return attrname in self.__dict__

    def __repr__(self):
        name = self.__class__.__name__
        return f"<{name}: {str(self.__dict__)}>"

    @property
    def exp_name(self):
        return f"{self.model_name}_{self.dataset_name}"

    @property
    def config_dict(self):
        return self.config

    @staticmethod
    def parse_encoder_dataset(dataset_l):
        for dataset_s in dataset_l:
            assert dataset_s in Config.encoder_datasets, f"Could not find {dataset_s}"
        return [Config.encoder_datasets[s]() for s in dataset_l]

    @staticmethod
    def parse_classifier_dataset(dataset_s):
        assert dataset_s in Config.classifier_datasets, f"Could not find {dataset_s}"
        return Config.classifier_datasets[dataset_s]

    @staticmethod
    def parse_model(model_dict):
        model_s = model_dict["class"]
        assert model_s in Config.models, f"Could not find {model_s}"

        if model_s == "CPAN":
            mod = model_dict["mod"]

            def model(dim_features, dim_target, config):
                return Config.models[model_s](dim_features, dim_target, config, mod)

            return model
        return Config.models[model_s]

    @staticmethod
    def parse_encoder_loss(loss_s):
        assert loss_s in Config.encoder_losses, f"Could not find {loss_s}"
        return Config.encoder_losses[loss_s]

    @staticmethod
    def parse_classifier_loss(loss_s):
        assert loss_s in Config.classifier_losses, f"Could not find {loss_s}"
        return Config.classifier_losses[loss_s]

    @staticmethod
    def parse_metrics(metrics_s):
        met = list()
        for s in metrics_s:
            assert s in Config.metrics, f"Could not find {s}"
            met.append(Config.metrics[s])
        return met

    @staticmethod
    def parse_optimizer(optim_s):
        assert optim_s in Config.optimizers, f"Could not find {optim_s}"
        return Config.optimizers[optim_s]

    @staticmethod
    def parse_scheduler(sched_dict):
        if sched_dict is None:
            return None

        sched_s = sched_dict["class"]
        args = sched_dict["args"]

        assert sched_s in Config.schedulers, f"Could not find {sched_s}"

        return lambda opt: Config.schedulers[sched_s](opt, **args)

    @staticmethod
    def parse_early_stopper(stopper_dict):
        if stopper_dict is None:
            return None

        stopper_s = stopper_dict["class"]
        args = stopper_dict["args"]

        assert stopper_s in Config.early_stoppers, f"Could not find {stopper_s}"

        return lambda: Config.early_stoppers[stopper_s](**args)

    @staticmethod
    def parse_encoder_early_stopper(stopper_dict):
        if stopper_dict is None:
            return None

        stopper_s = stopper_dict["class"]
        args = stopper_dict["args"]

        assert stopper_s in Config.early_stoppers, f"Could not find {stopper_s}"

        return lambda: Config.early_stoppers[stopper_s](**args)

    @staticmethod
    def parse_gradient_clipping(clip_dict):
        if clip_dict is None:
            return None
        args = clip_dict["args"]
        clipping = None if not args["use"] else args["value"]
        return clipping

    @classmethod
    def from_dict(cls, dict_obj):
        return Config(**dict_obj)


class Grid:
    """
    Specifies the configuration for multiple models.
    """

    def __init__(self, path_or_dict):
        self.configs_dict = read_config_file(path_or_dict)
        self.num_configs = 0  # must be computed by _create_grid
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, "No configurations available"
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        """ Generate hyper parameter grid from yaml file for hyper parameter
        tuning.
        """
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        """
        Takes a dictionary of key:list pairs and computes all possible
            permutations.
        :param configs_dict:
        :return: A dictionary generator
        """
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list
