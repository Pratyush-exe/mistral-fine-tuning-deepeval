from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback


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


def create_callback(dataset, trainer):
    import os

    os.environ["OPENAI_API_KEY"] = "api-key"

    eval_dataset = create_eval_dataset(dataset)
    metrics = create_metrics()
    callback = DeepEvalHuggingFaceCallback(
        evaluation_dataset=eval_dataset, metrics=metrics, trainer=trainer
    )

    return callback
