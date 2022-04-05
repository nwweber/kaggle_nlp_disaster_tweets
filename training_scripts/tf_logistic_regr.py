"""
some tf idf preprocessing and logistic regression classifier
"""
from os.path import join as pjoin
from pathlib import Path
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from mlflow import log_metric, log_param, set_experiment
from mlflow.sklearn import log_model
from sklearn import feature_extraction, linear_model
from sklearn.feature_extraction.text import TfidfTransformer

from mlflow import log_metrics, log_params


train_df = pd.read_csv(project_root / "kaggle_data/train.csv")
test_df = pd.read_csv(project_root / "kaggle_data/test.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])
# transform only on the test set
test_vectors = count_vectorizer.transform(test_df["text"])

tf_transformer = TfidfTransformer(use_idf=False).fit(train_vectors)
X_train_tf = tf_transformer.transform(train_vectors)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_vectors)

logi = linear_model.LogisticRegressionCV(penalty='l2', cv = 5, max_iter = 1000, scoring="f1", random_state = 54, refit = False)
logi.fit(X_train_tfidf, train_df["target"])

scores_logi = logi.score(X_train_tfidf, train_df["target"])
scores_logi

