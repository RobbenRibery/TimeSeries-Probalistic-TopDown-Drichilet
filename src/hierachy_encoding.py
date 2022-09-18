import pandas as pd 
import numpy as np 
from collections import defaultdict 

def hie_encoder(df,cols): 

    """
    input df here must follow the form of a hierachy lookup table 
    """
    base = 0
    for col in cols: 
        
        df[f"{col}_"] = df[col].astype('category').cat.codes + base

        base = df[f"{col}_"].max() + 1

    return df 

def get_parent_index(dic, parent_dic): 

    res = defaultdict(int)
    for k, v in dic.items(): 

        keys = k.split('_')
        
        parent_key = "_".join(keys[:-1])
        
        res[k]=(parent_dic[parent_key])

    return res

def get_children_index(dic): 

    """
    return the parent node name and index 
    """

    res = defaultdict(list)
    for k, v in dic.items(): 

        parent_key = '_'.join(k.split('_')[:-1])
        res[parent_key].append(v)

    if '' in res: 

        res['root'] = res.pop('')

    return res

    