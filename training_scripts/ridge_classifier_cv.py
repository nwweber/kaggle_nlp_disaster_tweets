"""
train ridge cv model
"""
import os
from pathlib import Path
import pandas as pd
from mlflow import log_metric, log_param, set_experiment, set_tracking_uri
from mlflow.sklearn import log_model
from sklearn import feature_extraction, linear_model


def get_project_root() -> Path:
    """
    tries to get path to project root from path to this script. if impossible
    will ask user for input.
    assumes that current file lives in direct sub-dir of project root, e.g.
    project_root/some_dir/this_file
    :return:
    :rtype:
    """
    try:
        # project root is one level up from where this file lives
        # first .parent removes filename + extension, second one goes up one dir
        project_root: Path = Path(__file__).parent.parent
    except NameError:
        # thrown if __file__ is not defined
        # seems like you're running this interactively. better make sure you
        # figure out paths by yourself
        # recommendation: start python process in root dir of this repository
        project_root: Path = Path(input("Please enter the path to the root folder of this project. Both absolute \n"
                                        "paths (like /user/home/....) or relative ones (like ./../project) work. \n"
                                        f"Your current working directory is: {os.getcwd()} \n"
                                        f"(enter '.' if this is already correct) \n"
                                        f"Your input: ")).absolute()
    return project_root


def main() -> None:
    """
    train and log model
    :return:
    :rtype:
    """
    project_root: Path = get_project_root()
    # track in 'mlruns' folder under project root directory
    set_tracking_uri(f"file://{project_root/'mlruns'}")
    # check this experiment out in mlflow ui by running this in a shell:
    # mlflow ui
    set_experiment("nlp_disaster_tweets")

    train_df = pd.read_csv(project_root/"kaggle_data/train.csv")
    count_vectorizer = feature_extraction.text.CountVectorizer()
    train_vectors = count_vectorizer.fit_transform(train_df["text"])

    clf_cv = linear_model.RidgeClassifierCV(
        cv=5, scoring="f1", alphas=(0.1, 1.0, 10, 100, 1000)
    )
    clf_cv.fit(train_vectors, train_df["target"])

    # log fixed and learned hyperparameters to mlflow
    for p, v in clf_cv.get_params().items():
        log_param(p, v)
    log_param("best_alpha", clf_cv.alpha_)
    # log f1 score for best value of alpha (see above) to mlflow
    log_metric("f1", clf_cv.best_score_)


if __name__ == "__main__":
    main()
