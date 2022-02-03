"""
official tutorial taken from
https://www.kaggle.com/philculliton/nlp-getting-started-tutorial

changed in some parts, e.g file names
"""
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("kaggle_data/train.csv")
test_df = pd.read_csv("kaggle_data/test.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()
# let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
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
    clf, train_vectors, train_df["target"], cv=3, scoring="f1"
)
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("kaggle_data/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv("tutorial/submission.csv", index=False)
