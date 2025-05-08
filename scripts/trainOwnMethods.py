import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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


def LogisticRegressionGradientDescent(df: pd.DataFrame,learning_rate: float = 0.01, n_iters: int = 1000):
    """
    Implementacja regresji logistycznej.
    
    Parametry:
    df - DataFrame zawierający dane
    learning_rate - współczynnik uczenia
    n_iters - liczba iteracji
    """

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    df = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
    
    X = df.drop(['Target'], axis=1)
    y = df['Target'].map({'Graduate': 1, 'Dropout': 0}).values

    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    

    X_train_b = np.c_[np.ones(X_train_processed.shape[0]), X_train_processed]
    X_val_b = np.c_[np.ones(X_val_processed.shape[0]), X_val_processed]
    X_test_b = np.c_[np.ones(X_test_processed.shape[0]), X_test_processed]


    

    theta = np.zeros(X_train_b.shape[1])
    best_theta = theta.copy()
    best_val_loss = float('inf')
    patience = 50 
    no_improvement = 0
    
    for iteration in range(n_iters):
        indices = np.random.randint(0, len(y_train), 32)
        X_batch = X_train_b[indices]
        y_batch = y_train[indices]
        
        h = sigmoid(X_batch.dot(theta))
        gradient = X_batch.T.dot(h - y_batch) / 32
        theta -= learning_rate * gradient
        
        if iteration % 10 == 0:
            h_val = sigmoid(X_val_b.dot(theta))
            val_loss = -np.mean(y_val * np.log(h_val + 1e-7) + (1 - y_val) * np.log(1 - h_val + 1e-7))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_theta = theta.copy()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break
    
    theta = best_theta
    
    y_pred_proba = sigmoid(X_test_b.dot(theta))
    y_pred = (y_pred_proba >= 0.5).astype(int)

    y_pred_train_proba = sigmoid(X_train_b.dot(theta))
    y_pred_train = (y_pred_train_proba >= 0.5).astype(int)

    y_pred_val_proba = sigmoid(X_val_b.dot(theta))
    y_pred_val = (y_pred_val_proba >= 0.5).astype(int)
    

    
    print("\nWyniki modelu regresji logistycznej:")
    print("\nZbiór treningowy")
    print(f"Dokładność: {accuracy_score(y_train,y_pred_train):.4f}")
    print(f"Precyzja: {precision_score(y_train,y_pred_train):.4f}")
    print(f"Czułość: {recall_score(y_train,y_pred_train):.4f}")
    print(f"F1-score: {f1_score(y_train,y_pred_train):.4f}")

    print("\nZbiór testowy")
    print(f"Dokładność: {accuracy_score(y_test,y_pred):.4f}")
    print(f"Precyzja: {precision_score(y_test,y_pred):.4f}")
    print(f"Czułość: {recall_score(y_test,y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test,y_pred):.4f}")

    print("\nZbiór walidacyjny")
    print(f"Dokładność: {accuracy_score(y_val,y_pred_val):.4f}")
    print(f"Precyzja: {precision_score(y_val,y_pred_val):.4f}")
    print(f"Czułość: {recall_score(y_val,y_pred_val):.4f}")
    print(f"F1-score: {f1_score(y_val,y_pred_val):.4f}")

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

    LinearRegressionClosedFormula(df,column)
    LogisticRegressionGradientDescent(df, learning_rate=0.01, n_iters=10000)
    LinearRegressionToCheck(df,column)
    