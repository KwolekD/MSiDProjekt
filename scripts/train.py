import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score, mean_squared_error,recall_score,precision_score,f1_score
import os
import pandas as pd


def firstTask(df: pd.DataFrame):
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
        # 'Regresja logistyczna l2': LogisticRegression(max_iter=10000, random_state=42,penalty='l2',solver='lbfgs'),
        # 'Regresja logistyczna l1': LogisticRegression(max_iter=10000, random_state=42,penalty='l1',solver='liblinear'),
        # 'Regresja logistyczna bez': LogisticRegression(max_iter=10000, random_state=42,penalty=None),
        'Drzewo decyzyjne': DecisionTreeClassifier(random_state=42),
        'Las losowy': RandomForestClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        try:
            pipeline, accuracy = evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor,X,y_binary)
            results[name] = accuracy

            # print("\nWagi cech:")
            # try:
            #     classifier = pipeline.named_steps['classifier']
            
            #     ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
            #     cat_feature_names = ohe.get_feature_names_out(categorical_features)

            #     all_feature_names = np.concatenate([cat_feature_names, numerical_features])
                
            #     coefficients = classifier.coef_.ravel()
            #     for name, coef in zip(all_feature_names, coefficients):
            #         print(f"{name}: {coef:.4f}")
            # except Exception as e:
            #     print(f"Nie udało się wyświetlić wag cech: {e}")
    
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
        
        params = {}

        if model_name == "Drzewo decyzyjne":
            params = {
                'classifier__max_depth': [4,5, 6, 8, 12, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }

        elif model_name == "Las losowy":
            params = {
                'classifier__n_estimators': [100,200,250],
                'classifier__max_depth': [8,10,13,16,20,None],
                'classifier__min_samples_split': [2, 3,5 ],
                'classifier__min_samples_leaf': [1, 2, 6]
            }
        elif model_name == "Maszyna wektorów nośnych":
            params = {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf'],
            }

        if params:
            grid_search = GridSearchCV(clf, param_grid=params, cv=3, n_jobs=-1)

            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_

            print(f"\nNajlepsze parametry dla {model_name}:")
            print(grid_search.best_params_)
            print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        else:
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

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        for train_index, val_index in skf.split(X, y_binary):

            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_test_fold = y_binary[train_index], y_binary[val_index]
            
            clf.fit(X_train_fold, y_train_fold)
            y_pred_fold = clf.predict(X_test_fold)
            acc = accuracy_score(y_test_fold, y_pred_fold)
            cv_scores.append(acc)
            print(f"Dokładność: {acc:.4f}")
        
        print(f"\nWyniki kross-walidacji (3-krotna):")
        print(f"Średnia dokładność: {np.mean(cv_scores):.4f}")
        print(f"Odchylenie standardowe: {np.std(cv_scores):.4f}")

        
        return clf, accuracy
        


if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        df: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else :
        print("First run loadData.py to load the dataset.")
        exit(1)

    firstTask(df)





    




    

    

    
