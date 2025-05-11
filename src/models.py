import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

def train_kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model


def train_mlp(X, y, epochs=50, batch_size=32):
    num_classes = len(np.unique(y))
    # one-hot encode targets
    y_cat = to_categorical(y, num_classes)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # fit with a small validation split to monitor overfitting
    history=model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1
    )
    model.history = history
    return model
