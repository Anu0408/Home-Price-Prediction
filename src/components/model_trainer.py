import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
import configparser

config = configparser.RawConfigParser()

class ModelTrainer:
    def __init__(self):
        self.config = config.read("C:/Users/anucv/OneDrive/Desktop/AI and ML training/HomePrice/src/config/config.ini")
        mlflow.set_experiment("house_price_prediction")

    def log_model(self, model_name, model, params, metrics):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)

    def reg_model(self, X_train, X_test, y_train, y_test, X, y, rs):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred)
        }
        self.log_model('linear_regression', lr, {}, metrics)
        return metrics

    def ridge_model(self, X, y, rs):
        alphas = np.logspace(-3, 3, 100)
        ridge = Ridge()
        pg = {"alpha": alphas}
        grid_search = GridSearchCV(ridge, pg, cv=5, scoring='r2')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        metrics = {
            'best_score': grid_search.best_score_,
        }
        self.log_model('ridge', best_model, grid_search.best_params_, metrics)

    def lasso_model(self, X, y, rs):
        lasso = Lasso()
        alphas = np.logspace(-3, 3, 100)
        pg = {"alpha": alphas}
        grid_search = GridSearchCV(lasso, pg, cv=5, scoring='r2')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        metrics = {
            'best_score': grid_search.best_score_,
        }
        self.log_model('lasso', best_model, grid_search.best_params_, metrics)

    def elasticnet_model(self, X, y, rs):
        l1_ratio = np.random.rand(20)
        elastic = ElasticNet()
        pg = {"alpha": np.linspace(0.1, 1, 5), "l1_ratio": l1_ratio}
        grid_search = GridSearchCV(elastic, pg, cv=5, scoring='r2')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        metrics = {
            'best_score': grid_search.best_score_,
        }
        self.log_model('elasticnet', best_model, grid_search.best_params_, metrics)

    def svr_model(self, X, y, rs):
        svr = SVR()
        pg = {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto'],
            "C": np.logspace(-3, 3, 10),
            "epsilon": np.linspace(.1, 1., 10)
        }
        grid_search = GridSearchCV(svr, pg, cv=5, scoring='r2')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        metrics = {
            'best_score': grid_search.best_score_,
        }
        self.log_model('svr', best_model, grid_search.best_params_, metrics)

    def rand_fr(self, X_train, X, y, rs):
        rfr = RandomForestRegressor(random_state=10)
        pg = {
            "n_estimators": [10, 20, 30, 50],
            "criterion": ["squared_error", "absolute_error", "poisson"],
            "max_depth": [2, 3, 4],
            "min_samples_split": range(2, 10),
            "min_samples_leaf": [2, 3],
            "max_features": range(4, X_train.shape[1] + 1)
        }
        grid_search = GridSearchCV(rfr, pg, cv=5, scoring='r2')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        metrics = {
            'best_score': grid_search.best_score_,
        }
        self.log_model('random_forest', best_model, grid_search.best_params_, metrics)

    def knn_model(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsRegressor(n_neighbors=20, weights="uniform")
        knn.fit(X_train, y_train)
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred)
        }
        self.log_model('knn', knn, {}, metrics)

    def final_model(self,model_type,X,y,rs,X_train, X_test, y_train, y_test):
        
        MODEL_DIR=config.get('DATA', 'model_dir')
        
        if model_type=='linear':
            #model = reg_model(X_train, X_test, y_train, y_test, X,y,rs)
            lr = LinearRegression()
            model=lr.fit(X_train, y_train)
            print("LINEAR REGRESSION MODEL : ")
            pickle.dump(model, open(MODEL_DIR + 'reg_model.pkl', 'wb'))
            return model

        elif model_type=='ridge':
            model = ridge_model(X,y,rs)
            print("RIDGE REGRESSION MODEL : ")
            pickle.dump(model, open(MODEL_DIR + 'ridge_model.pkl', 'wb'))

        elif model_type=='lasso':
            model = lasso_model(X,y,rs)
            print("LASSO REGRESSION MODEL : ")
            pickle.dump(model, open(MODEL_DIR + 'lasso_model.pkl', 'wb'))

        elif model_type=='elasticnet':
            model = elasticnet_model(X,y,rs)
            print("ELASTICNET MODEL : ")
            pickle.dump(model, open(MODEL_DIR +  'elastic_model.pkl', 'wb'))

        elif model_type=='svr':
            model = svr_model(X,y,rs)
            print("SVR MODEL : ")
            pickle.dump(model, open(MODEL_DIR + 'svr_model.pkl', 'wb'))    

        elif model_type=='random':
            #model = rand_fr(X_train,X,y,rs)
            print("RANDOM MODEL : ")
            #pickle.dump(model, open(MODEL_DIR + 'randomforest_model.pkl', 'wb'))

        elif model_type=='knn':
            model = knn_model(X_train,X_test,y_train,y_test)
            print("KNN MODEL : ")
            pickle.dump(model, open(MODEL_DIR + 'knn_model.pkl', 'wb'))

        #elif model_type=='xgboost':
        #    model = xgboost_model(X_train,y_train,X_test,y_test,X,y,rs)
        #    print("XGBOOST MODEL : ")
        #    pickle.dump(model, open(MODEL_DIR + 'xgboost_model.pkl', 'wb'))

        else:
            print("invalid input")
            
            
    def find_best_model(self, X, y):
        models = {
            'linear_regression': {
                'model': LinearRegression(),
                'parameters': {}
            },
            'lasso': {
                'model': Lasso(),
                'parameters': {
                    'alpha':  np.logspace(-4, 4, 100),
                    'selection': ['random', 'cyclic']
                }
            },
            'svr': {
                'model': SVR(),
                'parameters': {
                    'C': np.logspace(-3, 3, 10),
                    'gamma': ['scale', 'auto'],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'epsilon': np.linspace(0.01, 1.0, 10)
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'parameters': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(criterion='friedman_mse'),
                'parameters': {
                'n_estimators': [10, 50, 100, 200],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                }
            },
            'knn': {
                'model': KNeighborsRegressor(algorithm='auto'),
                'parameters': {
                'n_neighbors': [3, 5, 10, 15, 20],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # p=1 for Manhattan distance, p=2 for Euclidean distance
                }
            }
        }

        scores = []
        for model_name, model_params in models.items():
            gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False)
            gs.fit(X, y)
            scores.append({
                'model': model_name,
                'best_parameters': gs.best_params_,
                'score': gs.best_score_
            })

            # Log the model and metrics
            self.log_model(model_name, gs.best_estimator_, gs.best_params_, {'best_score': gs.best_score_})

        return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score'])



