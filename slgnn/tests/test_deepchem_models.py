import unittest

from ..models.deepchem_gcn_tf._base_model import Model


class ConcreteModel(Model):

    def fit_on_batch(self):
        pass

    def predict_on_batch(self):
        pass

    def reload(self):
        pass

    def save(self):
        pass

    def get_task_type(self):
        pass

    def get_num_tasks(self):
        pass


class TestDeepchemBaseModel(unittest.TestCase):

    def test_abc_methods(self):
        self.assertRaisesRegex(TypeError,
                               r"Can't instantiate abstract class",
                               Model)

    def test_concrete_model(self):
        ConcreteModel()  # Not raising TypeError with a concrete model
