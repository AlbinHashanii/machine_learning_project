from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def train_naive_bayes(X, y):
    model = GaussianNB().fit(X, y)
    return model

def train_linear_regression(X, y):
    model = LinearRegression().fit(X, y)
    return model

def train_lgbm(X, y):
    model = LGBMRegressor(random_state=42).fit(X, y)
    return model

def train_xgboost(X, y):
    model = XGBRegressor(random_state=42).fit(X, y)
    return model

def train_catboost(X, y):
    model = CatBoostRegressor(random_state=42, verbose=0).fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100).fit(X, y)
    return model
