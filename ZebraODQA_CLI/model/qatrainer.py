from transformers import Trainer


class QATrainer(Trainer):
    """
    Custom Question Answering trainer
    HuggingFace baseClass inheritance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def test_base_model_prefix(self):
        return self.model.base_model_prefix
