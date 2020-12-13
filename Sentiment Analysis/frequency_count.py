import pandas as pd
from collections import Counter
from itertools import chain

en_df = pd.read_csv("clean_train.csv")
# split words into lists
v = [s.split() for s in en_df['review'].astype(str).tolist()]
# compute global word frequency
c = Counter(chain.from_iterable(v))
# filter, join, and re-assign
en_df['review'] = [' '.join([j for j in i if c[j] > 2]) for i in v]

en_df.to_csv('clean_train.csv', index = False)