
from prophet import Prophet 
import pandas as pd 
import numpy as np

def prophrt_pre_processing(df, calender): 


    df = df.rename({"Unnamed: 0":"day_index"})
    df.columns = ["day_index",'sales_quantity']

    df = df.merge(calender[['d','date']], right_on='d', left_on='day_index', how = 'left')

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df = df[['date','sales_quantity']]
    df.columns = ['ds','y']

    return df 


def prophet_train(df, day_seasonality=True): 

    model = Prophet(daily_seasonality=day_seasonality )
    model.fit(df)

    return model 


def prophet_forecast(model,future): 

    future = model.make_future_dataframe(periods=future)
    forecasting = model.predict(future) 

    return forecasting 


def get_root_proportions(df , forecast_df, future ,name, sample_submission): 

    df = df.drop(columns = ['item_id','dept_id','cat_id','store_id','state_id'])
    df_values = df.values
    df_values = df_values[:,1:] 


    sum_by_day = df_values.sum(axis = 0)
    percentage_by_day = df_values/sum_by_day 

    historical_percentage = percentage_by_day.mean(axis = 1, keepdims=True ) 

    y_hat = forecast_df.tail(future)['yhat']
    y_hat = y_hat.values.reshape(-1,1) 

    leaf_forecast = np.dot(historical_percentage, np.transpose(y_hat))
    print(f"Leaft forecast matrix has shape:{leaf_forecast.shape}")

    result_float = pd.DataFrame(leaf_forecast, index= df['id'], columns = sample_submission.columns[1:] )
    result_float = result_float.astype(float)
    
    return result_float 


def post_processing(valid, eval, sample_submission, INTRESULT = False): 

    result = pd.concat((valid, eval), axis = 0)

    if INTRESULT:
        result = result.apply(np.round) 

    result = result.reset_index() 

    # validation
    for result_id, submission_id in zip(list(result['id']), list(sample_submission['id'])): 

        if result_id != submission_id: 
            print(result_id, submission_id)
            raise "Incorrect submmission index found"

    return result 

