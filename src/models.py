import numpy as np
import pandas as pd
from keras.src.layers import Dropout
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import regularizers
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras import Sequential, Input, layers, models
from sklearn.utils import compute_class_weight, class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, BatchNormalization
import torch
from sklearn.ensemble import ExtraTreesClassifier

def train_extra_trees(X, y):
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def train_naive_bayes(X, y):
    model = GaussianNB().fit(X, y)
    return model

def train_linear_regression(X, y):
    model = LinearRegression().fit(X, y)
    return model

def train_lgbm(X, y):
    return LGBMClassifier(random_state=42).fit(X, y)

def train_xgboost(X, y):
    return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss').fit(X, y)

def train_catboost(X, y):
    return CatBoostClassifier(random_state=42, verbose=0).fit(X, y)

def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100).fit(X, y)
    return model

def train_kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

def train_rnn(X, y, epochs=40, batch_size=32):

    X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

    n_samples, n_features = X_arr.shape
    X_seq = X_arr.reshape((n_samples, n_features, 1))

    num_classes = len(np.unique(y_arr))

    model = Sequential([
        Input(shape=(n_features, 1)),
        layers.LSTM(64, activation='tanh'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_seq, y_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    return model

def train_dense_nn(X, y, epochs=300, batch_size=256, learning_rate=1e-3):
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X.values.astype("float32"))
    y_arr = y.values.astype("int64")

    cw_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_arr),
        y=y_arr
    )
    cw = dict(enumerate(cw_vals))

    num_classes = len(np.unique(y_arr))
    model = Sequential([
        Dense(512, activation='relu',
              input_shape=(X_arr.shape[1],),
              kernel_regularizer=regularizers.L2(1e-4)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(256, activation='relu',
              kernel_regularizer=regularizers.L2(1e-4)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(128, activation='relu'),
        Dropout(0.1),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_arr, y_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=cw,
        verbose=1
    )

    return model, scaler




