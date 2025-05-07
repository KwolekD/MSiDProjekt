import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score, mean_squared_error,recall_score,precision_score,f1_score
import os
import pandas as pd


def firstTask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zakres na ocene 3.0 
    """

    df = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()

    X = df.drop(['Target'], axis=1)
    y = df['Target']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print("\nZmienne kategoryczne:", len(categorical_features))
    print("Zmienne numeryczne:", len(numerical_features))


    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    y_binary = y.map({'Graduate': 1, 'Dropout': 0}).values
    print("\nKlasy po binarnej transformacji:")
    print(pd.Series(y_binary).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    models = {
        'Regresja logistyczna': LogisticRegression(max_iter=1000, random_state=42),
        'Maszyna wektorów nośnych': SVC(random_state=42),
        'Drzewo decyzyjne': DecisionTreeClassifier(random_state=42,max_depth=6),
        'Las losowy': RandomForestClassifier(random_state=42,max_depth=9),
    }

    results = {}
    for name, model in models.items():
        try:
            pipeline, accuracy = evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor,X,y_binary)
            results[name] = accuracy
        except Exception as e:
            print(f"Błąd przy trenowaniu modelu {name}: {e}")

    if results:
        best_model = max(results, key=results.get)
        print(f"\nNajlepszy model: {best_model} z dokładnością {results[best_model]:.4f}")
    else:
        print("\nNie udało się wytrenować żadnego modelu.")
    

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor,X,y_binary):
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)

        y_pred_train = clf.predict(X_train)
        
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n--- {model_name} ---")
        print("\nZbiór treningowy")
        print(f"Dokładność: {accuracy_score(y_train,y_pred_train):.4f}")
        print(f"Precision: {precision_score(y_train,y_pred_train):.4f}")
        print(f"Recall: {recall_score(y_train,y_pred_train):.4f}")
        print(f"F1: {f1_score(y_train,y_pred_train):.4f}")

        print("\nZbiór testowy")
        print(f"Dokładność: {accuracy_score(y_test,y_pred):.4f}")
        print(f"Precision: {precision_score(y_test,y_pred):.4f}")
        print(f"Recall: {recall_score(y_test,y_pred):.4f}")
        print(f"F1: {f1_score(y_test,y_pred):.4f}")
        # print("\nRaport klasyfikacji:")
        # print(classification_report(y_test, y_pred, target_names=["Dropout","Graduate"]))
        
        cv_scores = cross_val_score(clf, X, y_binary, cv=5)
        print(f"\nWyniki kross-walidacji (5-krotna):")
        print(f"Średnia dokładność: {cv_scores.mean():.4f}")
        print(f"Odchylenie standardowe: {cv_scores.std():.4f}")
        
        return clf, accuracy

if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        df: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else :
        print("First run loadData.py to load the dataset.")
        exit(1)

    firstTask(df)





    




    

    

    
