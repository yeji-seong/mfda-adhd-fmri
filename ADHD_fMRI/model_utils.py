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


def align_funcdata(X, region_data, grid_points):
    n = len(X)
    aligned_funclist = []

    for num in range(n):
        start_time = time.time()

        aligned_df = pd.DataFrame(index=range(82), columns=range(grid_points))

        subject_data = X[num]

        for idx, region_id in enumerate(region_data["Region_ID"]):
            region_index = region_data[region_data["Region_ID"] == region_id].index[0]
            start = region_data.loc[:region_index - 1, "Count"].sum()
            end = region_data.loc[:region_index, "Count"].sum()

            region_signals = subject_data.iloc[start:end]

            reg_fdg = FDataGrid(region_signals.iloc[0].values.reshape(1, -1, 1),
                                grid_points=np.arange(grid_points))

            for i in range(1, len(region_signals)):
                fdg = FDataGrid(region_signals.iloc[i].values.reshape(1, -1, 1),
                                grid_points=np.arange(grid_points))
                reg_fdg = reg_fdg.concatenate(fdg)

            aligned_fdg = ElasticRegistration().fit_transform(reg_fdg)

            aligned_mean_df = pd.DataFrame(np.mean(aligned_fdg).data_matrix[0]).T

            aligned_df.iloc[idx] = aligned_mean_df.values[0]

        aligned_funclist.append(aligned_df)

        elapsed_min = round((time.time() - start_time) / 60)
        print(f"{num + 1}/{n} is done, Time: {elapsed_min} min")

    return aligned_funclist


def calc_curvelen(aligend_funclist):

    r = len(aligend_funclist[0])
    t = len(aligend_funclist)

    total_curve_df = pd.DataFrame(index=range(r), columns=range(t))

    for time_idx in range(t):
        current_df = aligend_funclist[time_idx]

        for region_idx, row in current_df.iterrows():
            diffs = np.abs(np.diff([0] + row.tolist()))
            total_curve_df.iloc[region_idx, time_idx] = sum(diffs)

    return total_curve_df.T


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
