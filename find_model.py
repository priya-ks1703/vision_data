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
        seed: int,
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

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = AutoML(seed=seed)
    # load the dataset and create a loader then pass it
    automl.fit(dataset_class)

    # save automl._model
    logger.info("Saving model to disk")
    with output_path.open("wb") as f:
        torch.save(automl._model, f)


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
        default=Path("model.pth"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './model.pth'."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
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
        f"Running dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        output_path=args.output_path,
        seed=args.seed,
    )
