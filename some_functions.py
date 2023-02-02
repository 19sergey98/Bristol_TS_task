def create_features_v1(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype("int64")
    
    # has_major_holiday_coming accounts for new year and may holidays 
    df['has_major_holiday_coming'] = 0
    df.loc[df['month'] == 12 , 'has_major_holiday_coming'] = 1
    df.loc[df['month'] == 5 , 'has_major_holiday_coming'] = 1
    return df

def get_monthly_data_prepared(df):
    monthly_data=df.groupby(['Store','MON_DATE'],as_index=False).sum()
    monthly_data.sort_values(by=['MON_DATE'])
    monthly_data.rename(columns={"Sales": "MON_SALES"},inplace=1)
    monthly_data=monthly_data.set_index("MON_DATE")
    data_with_features = create_features_v1(monthly_data)
    return data_with_features

def get_daily_data_prepared(df):
    daily_data = df.copy()
    daily_data.drop(columns='MON_DATE',inplace=True)
    daily_data=daily_data.set_index('Date')
    data_with_features = create_features_v1(daily_data)
    return data_with_features

def x_y_split_task1(df):
    FEATURES = ['Store','dayofyear','dayofmonth', 'dayofweek', 'year', 'quarter', 'month','has_major_holiday_coming','weekofyear']
    TARGET = 'MON_SALES'
    X = df[FEATURES]
    y = df[TARGET]
    return X,y

def x_y_split_task2(df):
    FEATURES = ['Store','dayofyear','dayofmonth', 'dayofweek', 'year', 'quarter', 'month','has_major_holiday_coming','weekofyear']
    TARGET = 'Sales'
    X = df[FEATURES]
    y = df[TARGET]
    return X,y

def get_score(predicted_values,true_values):
    wape=0
    true_sum=0
    
    if(len(predicted_values)!=len(true_values)):
        print("Размеры векторов не совпдают!!!!!!")
        return 1000000 # means predictions doesn't match expected size, sth wrong
    
    # get total sum
    for elem in true_values:
        true_sum=true_sum+abs(elem)
        
    for i in range(len(predicted_values)):
        wape = wape + abs(predicted_values[i]-true_values[i])
    
    wape = wape/true_sum # weight
    return wape