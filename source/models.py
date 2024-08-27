from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from interpret.glassbox import ExplainableBoostingRegressor

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

from source.metrics import *


def regression_tree(X_train, X_test, y_train, y_test, cv: int = 0, print_metrics: bool = False):
    reg_tree = DecisionTreeRegressor(random_state=42)
    reg_tree.fit(X_train, y_train)

    if cv == 0:
        y_pred = reg_tree.predict(X_test)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics = print_metrics, method = 'Regression Tree')

    else:
        y_pred = cross_val_predict(reg_tree, X_test, y_test, cv=cv)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics = print_metrics, method = 'Regression Tree Cross Val')

    return RMSE, MAE, reg_tree


def lasso_reg(X_train, X_test, y_train, y_test, alpha:float = .1, cv: int = 0, print_metrics: bool = False):
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train, y_train)

    if cv == 0:
        y_pred = lasso_reg.predict(X_test)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics = print_metrics, method = 'Lasso Regression')
    else:
        y_pred = cross_val_predict(lasso_reg, X_test, y_test, cv=cv)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics = print_metrics, method = 'Lasso Regression Cross Val')

    return RMSE, MAE, lasso_reg

def explainable_boosting(X_train, X_test, y_train, y_test, feature_names, print_metrics: bool = False, cv: int = 0):
    ebm = ExplainableBoostingRegressor(min_samples_leaf=1, 
                        feature_names=feature_names, 
                        binning='quantile_humanized', 
                        max_leaves=100, 
                        interactions=4,
                        outer_bags=32, 
                        inner_bags=0 )

    if cv == 0:
        ebm.fit(X_train, y_train)
        y_pred = ebm.predict(X_test)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics=print_metrics, method = 'Explainable Booting Machine')

    else:
        search = RandomizedSearchCV(ebm, scoring=metrics.mean_squared_error, cv=cv,
                            n_iter=10, param_distributions={'max_leaves': [4, 16, 64, 128, 256]},
                            refit=True, n_jobs=-1)
        search.fit(X_train, y_train)
        ebm = search.best_estimator_

        y_pred = ebm.predict(X_test)
        RMSE, MAE = rmse_mae(y_pred, y_test, print_metrics = print_metrics, method = 'Explainable Booting Machine Cross Val')

    return RMSE, MAE, ebm