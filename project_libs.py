from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#----------------------------------------------------------------------------------------------------------------------

def read_csv_to_dataframe(file_path):
    """
    Read a CSV file into a pandas DataFrame.
    Parameters:
    - file_path (str): Path to the CSV file to be read.
    Returns:
    - df (pandas.DataFrame): DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"Error occurred while reading the CSV file: {str(e)}")
        return None
#-------------------------------------------------------------------------------------------------------------------------------
def preprocess_missing_values(df):
    """
    Read a CSV file into a pandas DataFrame and preprocess it to handle missing values.
    Parameters:
    - file_path (str): Path to the CSV file to be read.
    Returns:
    - df (pandas.DataFrame): Preprocessed DataFrame containing the data from the CSV file,
                             with no missing values.
    """
    try:
        # Read the CSV file
        #df = pd.read_csv(file_path)
        # Preprocess the DataFrame to handle missing values
        # For numerical columns, fill missing values with the median
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        # For categorical columns, fill missing values with a placeholder string 'Unknown'
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna('Unknown', inplace=True)
        return df
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"Error occurred while reading the CSV file: {str(e)}")
        return None

#---------------------------------------------------------------------------------------------------------------------------------
def split_data(data, target_column, split_size=0.2, random_state=None):
    """
    Splits the data into features (X) and target (y), and further splits them into
    training and testing sets.

    Parameters:
    - data: Pandas DataFrame, input data
    - target_column: str, the name of the target column
    - split_size: float, the proportion of the data to include in the test split (default is 0.2)
    - random_state: int or None, seed for random number generation (default is None)

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets for features (X) and target (y)
    
    - # Example usage:
    # Assuming 'your_data' is your DataFrame and 'your_target_column' is the target column name
    X_train, X_test, y_train, y_test = split_data(your_data, 'your_target_column', split_size=0.2, random_state=42)
    """
    try:
        print(f'Data Frame shape {data.shape}')
        # Extract features (X) and target (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Perform the initial split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=random_state)
        print(f'{X_train.shape, X_test.shape, y_train.shape, y_test.shape}')
        return X_train, X_test, y_train, y_test
    except ValueError as ve:
        print(f"Error: {ve}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None
#----------------------------------------------------------------------------------------------------------------------------------------

def Feature_Encoding(df, features=None, encoding_method='one_hot'):
    """
    Hot encoding or label encoding for specified features.
    """
    if features is None:
        raise ValueError("List of features must be provided.")
    
    if encoding_method not in ['one_hot', 'label']:
        raise ValueError("Invalid encoding method. Please choose 'one_hot' or 'label'.")
    
    if encoding_method == 'one_hot':
        encoder = OneHotEncoder()
    else:
        encoder = LabelEncoder()
    
    # Select only the specified features from the DataFrame
    df_subset = df[features]
    
    if encoding_method == 'one_hot':
        encoded_data = encoder.fit_transform(df_subset)
        return pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(features))
    else:
        encoded_data = df_subset.apply(encoder.fit_transform)
        return encoded_data
    

#----------------------------------------------------------------------------------------------------------
def plot_correlation(data):
    """
    Plot correlation matrix using Seaborn.

    Parameters:
    - data: Pandas DataFrame, the input data for which to calculate and visualize the correlation.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap using Seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Show the plot
    plt.title('Correlation Matrix')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------

def create_pair_plot(data):
    """
    Create a pair plot of all columns in the given DataFrame.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame.

    Returns:
    - None
        Displays the pair plot.
    """
    sns.set(style="ticks")
    sns.pairplot(data)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------


