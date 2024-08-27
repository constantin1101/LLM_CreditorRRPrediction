from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse_mae(y_pred, y_test, print_metrics: bool = False, method: str = ''):
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MAE = mean_absolute_error(y_test, y_pred)
    if print_metrics:
        print(f'{method} RMSE: {RMSE}')
        print(f'{method} MAE: {MAE}')

    return RMSE, MAE