import os

import yaml


class Renamer:
    def __init__(self, required_fields=None):
        if required_fields is None:
            self.required_fields = [
                "data_ratio",
                "encoder_dataset",
                "classifier_dataset",
                "encoder_epochs",
                "mod",
            ]
        else:
            self.required_fields = required_fields

    def rename_result(self, path):
        conf_file = os.path.join(path, "configs.yml")
        try:
            configs = yaml.load(open(conf_file, "r"), Loader=yaml.FullLoader)
        except FileNotFoundError:
            return
        new_name = ""
        for key, value in configs.items():
            if key not in self.required_fields:
                continue
            if isinstance(value, list):
                value = value[0]
            if new_name == "":
                new_name += str(value)
            else:
                new_name += "_" + str(value)
        counter = 1
        while 1:
            try:
                os.rename(path, os.path.join(os.path.dirname(path), new_name))
                break
            except FileExistsError:
                counter += 1
                new_name += "_" + str(value) + "_" + str(counter)

    def rename_results(self, path):
        results = os.scandir(path)
        for res in results:
            if not res.is_dir():
                continue
            self.rename_result(res.path)
