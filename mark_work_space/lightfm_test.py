import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix,coo_matrix
from pandas.api.types import CategoricalDtype


train_path = "/Users/matianjun/Documents/kdd_competition/data"
test_path = "/Users/matianjun/Documents/kdd_competition/data"

def get_two_categories(current_stage:int)->(pd.CategoricalDtype,pd.CategoricalDtype):
    user_c_set = set()
    item_c_set = set()
    for c in range(0,current_stage+1):
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        user_c_set = user_c_set.union(set(list(click_train.user_id.unique())))
        item_c_set = item_c_set.union(set(list(click_train.item_id.unique())))

    user_c = CategoricalDtype(sorted(list(user_c_set)), ordered=True)
    item_c = CategoricalDtype(sorted(list(item_c_set)), ordered=True)
    return user_c,item_c

def get_two_c_per_phrase(df:pd.DataFrame)->(pd.CategoricalDtype,pd.CategoricalDtype):
    user_c = CategoricalDtype(sorted(df.user_id.unique()), ordered=True)
    item_c = CategoricalDtype(sorted(df.item_id.unique()), ordered=True)
    return user_c,item_c


def sample_recommendation(col,model, user_ids):

    for user_id in user_ids:

        scores = model.predict(user_id, col.values.copy())
        top_items = col.values.copy()[np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


def main():
    current_stage = 6
    model = LightFM(no_components=30)
    # train_user_c_all ,train_item_c_all = get_two_categories(current_stage)
    for c in range(0,current_stage+1):
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    
        train_user_c ,train_item_c = get_two_c_per_phrase(click_train)
        click_train["time"] = 1
        row = click_train.user_id.astype(train_user_c).cat.codes
        col = click_train.item_id.astype(train_item_c).cat.codes
        # parse_matrix = csr_matrix((click_train["time"], (row, col)), \
        parse_matrix = coo_matrix((click_train["time"],(row, col)), \
                            shape=(train_user_c.categories.size, train_item_c.categories.size))
        import ipdb; ipdb.set_trace()
        print("phrase: ",c," totall user: ",parse_matrix.shape[0]," totall item: ",parse_matrix.shape[1],"\n")

        model.fit(parse_matrix.copy(),epochs=20)

        sample_recommendation(col,model, [0,1000,300])

        # prediction = model.predict(4965,col.values.copy())
        # print(prediction)

    # row = click_train.user_id.astype(train_user_c_all).cat.codes
    # col = click_train.item_id.astype(train_item_c_all).cat.codes
    # sample_recommendation(col,model, [29898,3102,34122,13739])



if __name__ == "__main__":
    main()