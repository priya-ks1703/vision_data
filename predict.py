"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np
from automl.automl import AutoML
import argparse
import torch
import logging

from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset

logger = logging.getLogger(__name__)


def main(
        dataset: str,
        output_path: Path,
        model_path: Path
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case "skin_cancer":
            dataset_class = SkinCancerDataset
        case "final-test-dataset":
            dataset_class = SkinCancerDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info("Fitting AutoML")

    automl = AutoML(seed=42)

    # Load model
    model = torch.load(model_path)

    automl._model = model

    # Do the same for the test dataset
    test_preds, test_labels = automl.predict(dataset_class)

    # Write the predictions of X_test to disk
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # In case of running on the test data, also add the predictions.npy
    # to the correct location for autoevaluation.
    if dataset=="skin_cancer":
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # In case of running on the test data, also add the predictions.npy
    # to the correct location for autoevaluation.
    if dataset=="final-test-dataset":
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test split for dataset '{dataset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions", "skin_cancer", "final-test-dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model.pth"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './model.pth'."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Evaluating model at {args.model_path} on dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        output_path=args.output_path,
        model_path=args.model_path,
    )
