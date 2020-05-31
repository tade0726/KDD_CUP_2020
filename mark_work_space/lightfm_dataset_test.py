import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix, coo_matrix
from pandas.api.types import CategoricalDtype
from lightfm.data import Dataset

log = print
train_path = "/Users/mark/Documents/local_code/kdd_competition/data"
test_path = "/Users/mark/Documents/local_code/kdd_competition/data"
# train_path = "/Users/matianjun/Documents/kdd_competition/data"
# test_path = "/Users/matianjun/Documents/kdd_competition/data"


def main():
    current_stage = 6
    model = LightFM(no_components=30)
    dataset = Dataset()

    for c in range(0, current_stage + 1):
        click_train = pd.read_csv(
            train_path + "/underexpose_train_click-{}.csv".format(c),
            header=None,
            names=["user_id", "item_id", "time"],
        )
        click_test = pd.read_csv(
            test_path + "/underexpose_test_click-{}.csv".format(c),
            header=None,
            names=["user_id", "item_id", "time"],
        )
        dataset.fit_partial(click_train["user_id"], click_train["item_id"])
        num_users, num_items = dataset.interactions_shape()
        log('Num users: {}, num_items {}.'.format(num_users, num_items))


        # train_user_c, train_item_c = get_two_c_per_phrase(click_train)
        # click_train["time"] = 1
        # row = click_train.user_id.astype(train_user_c).cat.codes
        # col = click_train.item_id.astype(train_item_c).cat.codes
        # # parse_matrix = csr_matrix((click_train["time"], (row, col)), \
        # parse_matrix = coo_matrix(
        #     (click_train["time"], (row, col)),
        #     shape=(train_user_c.categories.size, train_item_c.categories.size),
        # )
        # import ipdb

        # ipdb.set_trace()
        # print(
        #     "phrase: ",
        #     c,
        #     " totall user: ",
        #     parse_matrix.shape[0],
        #     " totall item: ",
        #     parse_matrix.shape[1],
        #     "\n",
        # )

        # model.fit(parse_matrix.copy(), epochs=20)

        # sample_recommendation(col, model, [0, 1000, 300])

        # prediction = model.predict(4965,col.values.copy())
        # print(prediction)

    # row = click_train.user_id.astype(train_user_c_all).cat.codes
    # col = click_train.item_id.astype(train_item_c_all).cat.codes
    # sample_recommendation(col,model, [29898,3102,34122,13739])


if __name__ == "__main__":
    main()
