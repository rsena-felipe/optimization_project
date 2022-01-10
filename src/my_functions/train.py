import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_features_target(input_file, target_column, index_column, format_type = "csv"):
    """
    Divide a dataframe in features (X) and target (y)

    Arguments:
    input_file (str) -- Path to the csv or pickle file of the dataset (The csv separator should be ",")
    target_column (str) --  Name of the target column (y), all the columns that are not the target column are going to be features
    index_column (str) -- Index column of the dataset
    format_type (str) -- Format type of the save data (csv or pickle)

    Returns:
    X -- A numpy.ndarray of features
    y -- A numpy.ndarray of the target
    """
    
    if format_type == "csv":
        df = pd.read_csv(input_file)
    elif format_type == "pickle":
        df = pd.read_pickle(input_file)
    
    df = df.loc[:, df.columns != index_column] # Avoids using index column as training variable 
    X = df.loc[:, df.columns != target_column]
    y = df[target_column]

    return X, y


def plot_feature_importance(model, X, number_features = 15, title = "Visualizing Important Features"):
    """
    Plot feature importance of an ensemble model

    Arguments:
    model -- sklearn model ensemble model
    X -- Data where the model was trained
    number_features (int) -- number of features to see
    title (str) -- title of the plot

    Returns:
    Plot showing the importance of the features
    """
     
    # Visualizing Feature Importance
    feature_imp = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False).head(number_features)

    # Creating a bar plot
    sns.barplot(x=feature_imp["importance"], y=feature_imp.index, color="indianred")
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(title)
    plt.legend()
    plt.show()

    return plt









 