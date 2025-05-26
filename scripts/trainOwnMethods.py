import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score,precision_score,recall_score,f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def LinearRegressionClosedFormula(df: pd.DataFrame, numeric_column: str):
    """
    Implementacja regresji liniowej używającej zamkniętej formuły.
    
    Parametry:
    df - DataFrame zawierający dane
    numeric_column - nazwa kolumny numerycznej do przewidywania
    """

    categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_column not in numerical_features:
        raise ValueError(f"Kolumna {numeric_column} nie istnieje lub nie jest numeryczna")
    
    numerical_features.remove(numeric_column)

    X = df.drop([numeric_column], axis=1)
    y = df[numeric_column].values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_b = np.c_[np.ones(X_train_processed.shape[0]), X_train_processed]
    X_test_b = np.c_[np.ones(X_test_processed.shape[0]), X_test_processed]


    theta_best = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

    y_pred = X_test_b @ theta_best
    y_pred_train = X_train_b @ theta_best

    print(f"\nWyniki modelu regresji liniowej z zamkniętą formułą dla {numeric_column}:")
    print("\nZbiór treningowy")
    print(f"MSE: {mean_squared_error(y_train,y_pred_train):.4f}")
    print(f"MAE: {mean_absolute_error(y_train,y_pred_train):.4f}")
    print(f"R^2: {r2_score(y_train,y_pred_train):.4f}")

    print("\nZbiór testowy")
    print(f"MSE: {mean_squared_error(y_test,y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
    print(f"R^2: {r2_score(y_test,y_pred):.4f}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def LogisticRegressionGradientDescent(df: pd.DataFrame,learning_rate: float = 0.01, n_iters: int = 1000):
    """
    Implementacja regresji logistycznej.
    
    Parametry:
    df - DataFrame zawierający dane
    learning_rate - współczynnik uczenia
    n_iters - liczba iteracji
    """


    
    df = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
    
    X = df.drop(['Target'], axis=1)
    y = df['Target'].map({'Graduate': 1, 'Dropout': 0}).values

    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    

    X_train_b = np.c_[np.ones(X_train_processed.shape[0]), X_train_processed]
    X_test_b = np.c_[np.ones(X_test_processed.shape[0]), X_test_processed]
    
    theta = train_model(X_train_b, y_train, n_iters=n_iters, learning_rate=learning_rate)
    
    y_pred_proba = sigmoid(X_test_b.dot(theta))
    y_pred = (y_pred_proba >= 0.5).astype(int)

    y_pred_train_proba = sigmoid(X_train_b.dot(theta))
    y_pred_train = (y_pred_train_proba >= 0.5).astype(int)

    print(f"\nWyniki modelu regresji logistycznej:")
    print_classification_metrics(y_train, y_pred_train, "treningowy")
    print_classification_metrics(y_test,y_pred, "testowy")

    cv_scores = run_cross_validation(X, y, cv=3)

    print(f"\nWyniki walidacji krzyżowej (3-krotna):")
    print(f"Średnia dokładność: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


def run_cross_validation(X, y, cv=3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
        ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        

        X_train_b = np.c_[np.ones(X_train_processed.shape[0]), X_train_processed]
        X_test_b = np.c_[np.ones(X_test_processed.shape[0]), X_test_processed]

        theta = train_model(X_train_b, y_train,n_iters=10000)

        y_pred_proba = sigmoid(X_test_b.dot(theta))
        y_pred = (y_pred_proba >= 0.5).astype(int)

        score = accuracy_score(y_test, y_pred)
        print(f"Dokładność dla folda: {score:.4f}")
        cv_scores.append(score)

    return cv_scores

def train_model(X_train,y_train,n_iters=1000, learning_rate=0.01,lambda_reg=0.01):
    theta = np.zeros(X_train.shape[1])
    
    for iteration in range(n_iters):
        indices = np.random.randint(0, len(y_train), 32)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        h = sigmoid(X_batch.dot(theta))
        gradient = X_batch.T.dot(h - y_batch) / 32

        l2 = lambda_reg * theta * 0
        l1 = lambda_reg * np.sign(theta) * 1
        regularization = (l1+l2)/32
        
        regularization[0] = 0

        theta -= learning_rate * (gradient+regularization)
    return theta

def print_classification_metrics(y_true,y_pred,set_name):
    print(f"\nZbiór {set_name}")
    print(f"Dokładność: {accuracy_score(y_true,y_pred):.4f}")
    print(f"Precyzja: {precision_score(y_true,y_pred):.4f}")
    print(f"Czułość: {recall_score(y_true,y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true,y_pred):.4f}")


def LinearRegressionToCheck(df: pd.DataFrame, numeric_column: str):
    categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_column not in numerical_features:
        raise ValueError(f"Kolumna {numeric_column} nie istnieje lub nie jest numeryczna")
    
    numerical_features.remove(numeric_column)

    X = df.drop([numeric_column], axis=1)
    y = df[numeric_column].values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)



    model = LinearRegression()

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_pred_train = model.predict(X_train_processed)

    print(f"\nWyniki modelu regresji liniowej dla {numeric_column}:")
    print("\nZbiór treningowy")
    print(f"MSE: {mean_squared_error(y_train,y_pred_train):.4f}")
    print(f"MAE: {mean_absolute_error(y_train,y_pred_train):.4f}")
    print(f"R^2: {r2_score(y_train,y_pred_train):.4f}")

    print("\nZbiór testowy")
    print(f"MSE: {mean_squared_error(y_test,y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
    print(f"R^2: {r2_score(y_test,y_pred):.4f}")



if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        df: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else :
        print("First run loadData.py to load the dataset.")
        exit(1)


    column = "Curricular units 1st sem (grade)"

    # LinearRegressionClosedFormula(df,column)
    LogisticRegressionGradientDescent(df, learning_rate=0.01, n_iters=10000)
    # LinearRegressionToCheck(df,column)
    