from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback

from transformers import (
    TrainerCallback,
    ProgressCallback,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)


def create_goldens(dataset):
    goldens = []
    for i in range(len(dataset)):
        golden = Golden(input=dataset[i]["prompt"], actualOutput=dataset[i]["chosen"])
        goldens.append(golden)

    return goldens


def create_metrics():
    bias_metric = BiasMetric(threshold=0.5)
    toxicity_metric = ToxicityMetric(threshold=0.5)

    return [bias_metric, toxicity_metric]


def create_eval_dataset(dataset):
    goldens = create_goldens(dataset)
    dataset = EvaluationDataset(goldens=goldens)
    return dataset


def custom_on_log(
    self,
    args: TrainingArguments,
    state: TrainerState,
    control: TrainerControl,
    **kwargs,
):
    """
    Event triggered after logging the last logs.
    """
    if self.show_table and len(state.log_history) <= self.trainer.args.num_train_epochs:
        self.rich_manager.advance_progress()

        self.rich_manager.change_spinner_text(self.task_descriptions["evaluate"])

        scores = self._calculate_metric_scores()
        self.deepeval_metric_history.append(scores)
        self.deepeval_metric_history[-1].update(state.log_history[-1])

        print(self.deepeval_metric_history)

        import json

        f = open("file.txt", "w")
        f.write(json.dumps(self.deepeval_metric_history))

        self.rich_manager.change_spinner_text(self.task_descriptions["training"])
        columns = self._generate_table()
        self.rich_manager.update(columns)


def create_callback(dataset, trainer, tokenizer):
    import os

    os.environ["OPENAI_API_KEY"] = "api-key"

    eval_dataset = create_eval_dataset(dataset)
    metrics = create_metrics()
    callback = DeepEvalHuggingFaceCallback(
        evaluation_dataset=eval_dataset,
        metrics=metrics,
        trainer=trainer,
        show_table=True,
        tokenizer_args=tokenizer.tokenizer_args,
    )

    callback.on_log = custom_on_log.__get__(callback, DeepEvalHuggingFaceCallback)

    return callback
