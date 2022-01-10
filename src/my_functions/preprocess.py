import pandas as pd

def preprocess_original(input_file, output_file, delete_columns = ["Name", "Ticket", "Cabin"], drop_na = True):
    """
    Preprocess a raw dataframe and delete one or more columns and drops NA values.

    Arguments:
    input_file (str) -- Path to the csv file of the titanic dataset (The csv separator should be ";").
    output_file (str) -- Path where the new csv of the new process data is going to be saved.
    delete_columns -- List of colums to delete of the dataframe.
    drop_na -- Set True to drop all NA values.

    Returns:
    A .csv of the processed dataframe in the specified location. 
    """

    data = pd.read_csv(input_file, sep = ";")
    
    data.drop(delete_columns, axis = 1, inplace=True) 
    
    if drop_na == True:
        data.dropna(inplace=True)
    else:
        pass
    
    data.to_csv(output_file, index=False)

    