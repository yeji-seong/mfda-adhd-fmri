import numpy as np

from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def prep_data(X, y, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    spca = SparsePCA(n_components=20, alpha=1)
    spca.fit(X_train)
    
    X_train_spca = spca.transform(X_train)
    X_test_spca = spca.transform(X_test)
    
    return X_train_spca, X_test_spca, y_train, y_test


def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }


def train(model, param_grid, X_train, y_train, cv=10):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
