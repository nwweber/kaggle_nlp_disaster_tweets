"""
official tutorial taken from
https://www.kaggle.com/philculliton/nlp-getting-started-tutorial

changed in some parts, e.g file names
"""
from mlflow import log_metric, log_param, log_artifacts, set_experiment
from pathlib import Path
from os.path import join as pjoin
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

set_experiment('nlp_disaster_tweets')

train_df = pd.read_csv("kaggle_data/train.csv")
test_df = pd.read_csv("kaggle_data/test.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df["text"])
# note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors -
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])

# Our vectors are really big, so we want to push our model's weights
# toward 0 without completely discounting different words - ridge regression
# is a good way to do this.
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(
    clf, train_vectors, train_df["target"], cv=5, scoring="f1"
)
clf.fit(train_vectors, train_df["target"])

for p, v in clf.get_params().items():
    log_param(p, v)
log_metric('f1', scores.mean())
from mlflow.sklearn import log_model
log_model(clf, 'ridge_classififer')

# clf_cv = linear_model.RidgeClassifierCV(cv=5, scoring='f1', alphas=(0.1, 1.0, 10, 100, 1000))
# clf_cv.fit(train_vectors, train_df['target'])
# print(clf_cv.best_score_)
# print(clf_cv.alpha_)

sample_submission = pd.read_csv("kaggle_data/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
out_dir: str = "output"
Path(out_dir).mkdir(parents=True, exist_ok=True)
sample_submission.to_csv(pjoin(out_dir, "submission.csv"), index=False)
