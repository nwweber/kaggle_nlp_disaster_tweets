"""
train ridge cv model
"""
from os.path import join as pjoin
from pathlib import Path
import pandas as pd
from mlflow import log_metric, log_param, set_experiment
from mlflow.sklearn import log_model
from sklearn import feature_extraction, linear_model


def main() -> None:
    """
    train and log model
    :return:
    :rtype:
    """
    # check this experiment out in mlflow ui by running this in a shell:
    # mlflow ui
    set_experiment("nlp_disaster_tweets")
    project_root: Path = Path("./")
    try:
        # project root is one level up from where this file lives
        project_root: Path = Path(__file__).parent
    except NameError:
        # thrown if __file__ is not defined
        # seems like you're running this interactively. better make sure you
        # figure out paths by yourself
        # recommendation: start python process in root dir of this repository
        pass

    train_df = pd.read_csv(project_root / "kaggle_data/train.csv")
    test_df = pd.read_csv(project_root / "kaggle_data/test.csv")

    count_vectorizer = feature_extraction.text.CountVectorizer()
    train_vectors = count_vectorizer.fit_transform(train_df["text"])
    test_vectors = count_vectorizer.transform(test_df["text"])

    clf_cv = linear_model.RidgeClassifierCV(
        cv=5, scoring="f1", alphas=(0.1, 1.0, 10, 100, 1000)
    )
    clf_cv.fit(train_vectors, train_df["target"])
    for p, v in clf_cv.get_params().items():
        log_param(p, v)
    log_param("best_alpha", clf_cv.alpha_)
    log_metric("f1", clf_cv.best_score_)
    log_model(clf_cv, "ridge_classififer_cv")
    print(clf_cv.best_score_)
    print(clf_cv.alpha_)

    sample_submission = pd.read_csv("kaggle_data/sample_submission.csv")
    sample_submission["target"] = clf_cv.predict(test_vectors)
    out_dir: str = "output"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sample_submission.to_csv(pjoin(out_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
