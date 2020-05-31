import numpy as np

from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

print(repr(data["train"]))
print(repr(data["test"]))

import ipdb

ipdb.set_trace()
