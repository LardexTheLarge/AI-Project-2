from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
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

def Feature_Encoding(df, target_column, features=None, encoding_method='one_hot'):
    x_train, x_test, y_train, y_test = libs.split_data(df,target_column)
    """
    Hot encoding or label encoding for specified features.
    """
    if features is None:
        raise ValueError("List of features must be provided.")
    if encoding_method not in ['one_hot', 'label']:
        raise ValueError("Invalid encoding method. Please choose 'one_hot' or 'label'.")
    if encoding_method == 'one_hot':
        encoder = OneHotEncoder()
        for col in features:
            # Fit and transform on training data
            x_train[col]= encoder.fit_transform(x_train[col])
            # Transform on test data
            x_test[col] = encoder.fit_transform(x_test[col])
    else:
        encoder = LabelEncoder()
        for col in features:
            x_train[col] = encoder.fit_transform(x_train[col])
            x_test[col] = encoder.fit_transform(x_test[col])
    return x_train, x_test, y_train, y_test
    
feature = ['city','statezip']
x_train, x_test, y_train, y_test = Feature_Encoding(df, 'price', features=feature, encoding_method='label')

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
def run_ml_pipeline(X_train, X_test, y_train, y_test, models, use_cross_validation=True):
    """
    Run a pipeline of machine learning models on preprocessed data.

    Parameters:
    - X_train: Preprocessed training features
    - X_test: Preprocessed test features
    - y_train: Training target variable
    - y_test: Test target variable
    - models: List of tuples (model_name, model_instance, model_parameters)
    - use_cross_validation: Whether to use cross-validation or model scoring
    """

    score_test = []
    score_training = []
    model_names = []

    for model_name, model, params in models:
        # Create a pipeline for each model
        pipeline = Pipeline([
            ('model', model(**params))
        ])

        if use_cross_validation:
            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"Model: {model_name}")
            print(f"Mean Accuracy: {mean_score:.2f}")
            print(f"Standard Deviation: {std_score}")
            print("-" * 40)
            score_training.append(mean_score)
            score_test.append(np.nan)  # Cross-validation doesn't provide test scores
        else:
            # Fit the model and compute scores on training and test sets
            train_score = pipeline.fit(X_train, y_train).score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            print(f"Model: {model_name}")
            print(f"Training Score: {train_score:.2f}")
            print(f"Test Score: {test_score:.2f}")
            print("-" * 40)
            score_training.append(train_score)
            score_test.append(test_score)

        model_names.append(model_name)

    result = pd.DataFrame({'Model Name': model_names, 'Training Score': score_training, 'Test Score': score_test})
    print(result)

# Define the list of models to run in the pipeline
models = [
    ('Random Forest', RandomForestRegressor, {'n_estimators': 100}),
    ('Gradient Boosting', GradientBoostingRegressor, {'n_estimators': 100}),
    ('Ridge Regression', Ridge, {'alpha': 1.0}),
    ('Lasso Regression', Lasso, {'alpha': 1.0}),
    ('ElasticNet Regression', ElasticNet, {'alpha': 1.0, 'l1_ratio': 0.5}),
    ('Decision Tree', DecisionTreeRegressor, {'max_depth': 5}),
    ('Extra Trees', ExtraTreesRegressor, {'n_estimators': 100}),
    ('KNN', KNeighborsRegressor, {'n_neighbors': 5}),
    ('Gaussian Process', GaussianProcessRegressor, {})
]

run_ml_pipeline(X_train_scaled, X_test_scaled, y_train, y_test, models, use_cross_validation=True)

