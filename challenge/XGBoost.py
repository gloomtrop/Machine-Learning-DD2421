from xgboost import XGBRegressor

def training(train_X, train_y):
    my_model = XGBRegressor()
    my_model.fit(train_X, train_y, verbose=False)
    return my_model