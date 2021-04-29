from abc import ABC
import torch
from datasets import load_metric

class Validation(ABC):
    def __init__(self):
        self.metric = None

    def score_batch(self, model_predictions, gold_references):
        self.metric.add_batch(predictions=model_predictions, references=gold_references)
        return self.metric.compute()

    def score(self, model_prediction, gold_reference):
        self.metric.add(prediction=model_prediction, reference=gold_reference)
        return self.metric.compute()

class Accuracy(Validation):
    def __init__(self):
        super().__init__()
        self.metric = load_metric('accuracy', cache_dir=config_cache_dir)

class MedQA(Validation):
    def __init__(self):
        super().__init__()
        self.metric = load_metric('medqa', cache_dir=config_cache_dir)

class Squad(Validation):
    def __init__(self):
        super().__init__()
        self.metric = load_metric('squad', cache_dir=config_cache_dir)