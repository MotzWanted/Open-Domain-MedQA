from datasets import load_dataset

class MedQA(TrainTestDataset):
    def __init__(self, percent_of_data_to_keep=0.01):
        super().__init__()

        dataset = load_dataset(
            'json', 
            data_files={'train': os.path.join(gdrive_data_medqa,"questions/train/phrases_train.jsonl"),
                        'test': os.path.join(gdrive_data_medqa,"questions/test/phrases_test.jsonl")} ,
            block_size=40<<20, 
            split='train[:18]',
            cache_dir=config_cache_dir)

        dataset["train"] = self.select_part(
            dataset["train"], percent_of_data_to_keep
        )

        dataset["validation"] = self.select_part(
            dataset["validation"], percent_of_data_to_keep
        )

        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answers": data["answer"]
            },
            remove_columns=dataset.column_names["train"],
            num_proc=config_max_proc_to_use,
        )

        print(len(dataset["train"]))

        dataset = self.filter(dataset)

        print(len(dataset["train"]))

        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answers": self.check_answers(data["answer"])
            },
            remove_columns=dataset.column_names["train"],
            num_proc=config_max_proc_to_use
        )

        self.dataset = self.filter(dataset)

        print(len(self.dataset["train"]))