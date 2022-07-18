import pandas as pd 

def export_leaf_root(df,name):

    ts_leaf = df.groupby('id').sum()[df.columns[6:]]
    ts_leaf.to_csv(f'../data/{name}_leaf.csv')

    ts_root = ts_leaf.sum(axis = 0)
    ts_root = pd.DataFrame(ts_root,)
    ts_root.columns = ['sales_quantity']
    ts_root.to_csv(f'../data/{name}_root.csv')

    return ts_leaf, ts_root
