import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras import Sequential, Input, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, BatchNormalization


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
    # ensure numpy arrays
    X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

    n_samples, n_features = X_arr.shape
    # reshape to (samples, timesteps, features_per_step=1)
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

def train_dbn(X, y, epochs=20, batch_size=128):
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    unique_classes = np.unique(y_arr)
    class_map = {label: idx for idx, label in enumerate(unique_classes)}
    y_arr = np.array([class_map[label] for label in y_arr])
    num_classes = len(unique_classes)

    model = Sequential([
        Input(shape=(X_arr.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_arr), y=y_arr)
    class_weight_dict = dict(enumerate(class_weights))

    model.fit(
        X_arr, y_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict,
        verbose=1
    )

    return model
