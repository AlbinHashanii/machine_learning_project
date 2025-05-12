from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras import layers, models

def build_autoencoder(X_train):
    input_dim = X_train.shape[1]
    
    # Define Autoencoder architecture
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(32, activation='relu')(encoded)  # Bottleneck layer
    
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)  # Reconstructed output
    
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)  # Encoder model
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder

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
