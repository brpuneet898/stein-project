# scripts/data_utils.py
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_covtype_100pc(random_state=0, n_components=100, subset=None):
    X, y = fetch_covtype(return_X_y=True)
    y_bin = (y == 1).astype(int)

    if subset is not None:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=subset, replace=False)
        X = X[idx]
        y_bin = y_bin[idx]

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Adjust n_components if needed
    max_components = min(Xs.shape[1], n_components)
    if n_components > max_components:
        print(f"[data_utils] Requested n_components={n_components}, "
              f"but data has only {Xs.shape[1]} features. Using n_components={max_components}.")
        n_components = max_components

    # PCA
    if n_components < Xs.shape[1]:
        pca = PCA(n_components=n_components, random_state=random_state)
        Xp = pca.fit_transform(Xs)
    else:
        pca = None
        Xp = Xs  # no dimensionality reduction

    X_train, X_test, y_train, y_test = train_test_split(
        Xp, y_bin, test_size=0.2, random_state=random_state, stratify=y_bin
    )
    return X_train, X_test, y_train, y_test, scaler, pca