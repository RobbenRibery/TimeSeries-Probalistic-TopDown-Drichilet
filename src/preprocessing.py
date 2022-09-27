from ast import For
import pandas as pd 
import numpy as np 
import hierachy_encoding 

def get_parent_sales(data : pd.DataFrame): 

    """
    input:valudation / evaluaiton dataset 
    """

    return data.sum(axis=0)[6:].values.reshape(-1,1)


def get_child_proportions(data: pd.DataFrame, yp: pd.DataFrame): 

    """
    input:validation / evaluation dataset 
    """

    return data.groupby('cat_id').sum().values/yp.T

def get_embedding_input(ac:np.ndarray, level:str = None, cat_to_dix:dict = None,): 

    """
    input: 
    """

    ec = np.array(hierachy_encoding.get_children_index(cat_to_dix)[level]).reshape(-1,1)
    ec = np.repeat(ec, ac.shape[1], axis=1)

    return ec 


def get_input(ec:np.ndarray, ac:np.ndarray, yp:pd.DataFrame): 

    if ac.shape[0] == ec.shape[0]: 

        input = np.empty(
            
            (ac.shape[0], ac.shape[1], 3), 
        
            )
        print(input.shape)

        for c in range(input.shape[0]): 
            print(c)
            # print(yp.T.shape)
            # print(ac[c].reshape(-1,1).shape)
            # print(ec[c].reshape(-1,1).shape)
            input[c] = np.concatenate(
                [
                    ac[c].reshape(-1,1), 
                    yp, 
                    ## ---- PLACE HOLDER FOR COVARIATES X ---- ## 
                    ## ---- PLACE HOLDER FOR COVARIATES X ---- ## 
                    ec[c].reshape(-1,1),
                ], 
                axis = 1 
            )
            print(input[c].shape)    

    else: 
        raise "size of children in embedding does not agree with size of the children in proportions"

    return input 


def get_time_batched_arrays(History: int, Forward: int, input:np.ndarray):

    number_observations = input.shape[1] - (History + Forward) + 1

    input_time_batched = np.empty(
        (
            number_observations,input.shape[0], 
            History + Forward, 
            input.shape[-1]
        )
    )

    for i in range(number_observations):
        input_time_batched[i] = np.array(input[:, i:i + History + Forward, :])

    print("-------- X, y split ------------")
    input_array = np.empty((
        number_observations,
        input.shape[0],
        History,
        input.shape[-1])
    )

    target_array = np.empty((
        number_observations,
        input.shape[0],
        Forward,
        1)
    )
    for i in range(input_time_batched.shape[0]):

        input_array[i] = input_time_batched[i, :, :History, :]

        #print(input_array[i,0,-1,0])

        target_2d = input_time_batched[i, :, History:, 0]
        
        target_array[i] = target_2d.reshape(
            target_2d.shape[0], 
            target_2d.shape[1], 
            1
        )

    print(f"X input shape is {input_array.shape}")
    print(f"y input shape is {target_array.shape}")

    return input_array, target_array


def pre_processing_main(
    data: pd.DataFrame, 
    cat_to_index: dict,
    History: int,
    Forward: int, 
    ): 
    yp = get_parent_sales(data)
    ac = get_child_proportions(data, yp)
    ec = get_embedding_input(ac, level = 'root', cat_to_dix=cat_to_index)

    input = get_input(ec, ac, yp)

    input_array, target_array = get_time_batched_arrays(History, Forward, input)


    return input_array, target_array







