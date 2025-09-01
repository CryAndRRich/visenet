def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.timestamp >= start) & (df.timestamp < end)]
    data=data.sort_values(['timestamp','ticker'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.timestamp.factorize()[0]
    return data
