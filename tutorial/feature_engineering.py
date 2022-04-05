import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import join as pjoin
from pathlib import Path
from sklearn import feature_extraction, linear_model, model_selection

project_root: Path = Path("./")

train_df = pd.read_csv(project_root / "kaggle_data/train.csv")
test_df = pd.read_csv(project_root / "kaggle_data/test.csv")

hash = train_df['text'].str.extractall(r'(#\w+)').reset_index()

hash.groupby('level_0').match.max().plot(kind = 'hist')

plt.imshow(img.reshape((28, 28)))
plt.show()