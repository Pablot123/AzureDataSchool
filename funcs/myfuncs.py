import pandas as pd

def cln_data(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Delete the Fligh column
    input: dataset
    output: pandas dataframe withiout Flight
    '''
    clean_df = data.drop('Flight', axis=1, inplace=False)
    return clean_df

