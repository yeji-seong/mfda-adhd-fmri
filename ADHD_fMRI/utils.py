import numpy as np
import pandas as pd
import time
from skfda import FDataGrid
from skfda.preprocessing.registration import ElasticRegistration
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def align_funcdata(funcs, region, t_grid):
    n = len(funcs)

    aligned_funcs = []

    for nn in range(n):
        start_time = time.time()

        aligned_df = pd.DataFrame(index=range(len(region)), columns=range(t_grid))

        sub_funcs = funcs[nn]

        for ii, region_id in enumerate(region["Region_ID"]):
            idx = region.index[region["Region_ID"] == region_id][0]

            start = region.loc[:idx - 1, "Count"].sum()
            end = region.loc[:idx, "Count"].sum()

            region_funcs = sub_funcs.iloc[start:end]
            region_fdg = FDataGrid(region_funcs.iloc[0].values.reshape(1, -1, 1),
                                   grid_points=np.arange(t_grid))
            for rr in range(1, len(region_funcs)):
                fdg = FDataGrid(region_funcs.iloc[rr].values.reshape(1, -1, 1),
                                grid_points=np.arange(t_grid))
                region_fdg = region_fdg.concatenate(fdg)

            aligned_fdg = ElasticRegistration().fit_transform(region_fdg)
            aligned_mean_df = pd.DataFrame(np.mean(aligned_fdg).data_matrix[0]).T
            aligned_df.iloc[ii] = aligned_mean_df.values[0]

        aligned_funcs.append(aligned_df)

        execution_time = round((time.time() - start_time) / 60)
        print(f"{nn + 1}/{n} is done, Time: {execution_time} min")

    return aligned_funcs


def calc_curvelen(aligned_funcs):
    r = len(aligned_funcs[0])
    t = len(aligned_funcs)

    curvelen = pd.DataFrame(index=range(r), columns=range(t))

    for tt in range(t):
        aligned_fun = aligned_funcs[tt].to_numpy()
        diffs = np.abs(np.diff(aligned_fun, axis=1))
        lengths = diffs.sum(axis=1)

        for rr in range(r):
            curvelen.iloc[rr, tt] = lengths[rr]

    return curvelen.T


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
