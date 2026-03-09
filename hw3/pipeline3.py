import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cuml.linear_model import LogisticRegression as cuLogisticRegression


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*No categorical columns found.*")

results_store = []


def to_pandas_df(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if hasattr(df, "to_pandas"):
        return df.to_pandas().copy()
    return pd.DataFrame(df).copy()


def to_numpy(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, pd.Series):
        return x.to_numpy()

    mod = type(x).__module__

    if mod.startswith("cupy"):
        return x.get()

    if mod.startswith("cudf"):
        return x.to_pandas().to_numpy()

    return np.asarray(x)


def unique_list(values):
    return list(dict.fromkeys(values))

def clone_config(cfg):
    return {
        "target_cols": list(cfg.get("target_cols", [])),
        "ohe_cols": list(cfg.get("ohe_cols", [])),
        "numeric_cols": list(cfg.get("numeric_cols", [])),
        "mmr_transform": cfg.get("mmr_transform", "raw"),
    }

def add_mmr_to_config(base_cfg, mmr_transform="raw"):
    cfg = clone_config(base_cfg)
    cfg["numeric_cols"] = unique_list(cfg["numeric_cols"] + ["avg_mmr", "mmr_missing"])
    cfg["mmr_transform"] = mmr_transform
    return cfg


def make_submission_path(base_path, suffix):
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".csv"
    return f"{root}_{suffix}{ext}"



def gini_score(y_true, y_score):
    y_true = to_numpy(y_true).reshape(-1)
    y_score = to_numpy(y_score).reshape(-1)
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def extract_date_features(df):
    df["date"] = pd.to_datetime(df["date"])

    # df["day"] = df["date"].dt.day.astype(str)
    # df["weekday"] = df["date"].dt.weekday.astype(str)
    df["month"] = df["date"].dt.month.astype(str)
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(str)

    return df


def fit_mmr_fill_stats(df):
    group_means = df.groupby(["region", "game_mode"])["avg_mmr"].mean()
    region_means = df.groupby("region")["avg_mmr"].mean()
    global_mean = df["avg_mmr"].mean()

    return {
        "group_means": group_means,
        "region_means": region_means,
        "global_mean": global_mean,
    }


def transform_mmr_fill(df, stats):
    if "avg_mmr" not in df.columns:
        return df

    df["mmr_missing"] = df["avg_mmr"].isna().astype(int)

    if "region" in df.columns and "game_mode" in df.columns:
        pair_index = pd.MultiIndex.from_frame(df[["region", "game_mode"]])
        pair_fill = pd.Series(pair_index.map(stats["group_means"]), index=df.index)

        df["avg_mmr"] = df["avg_mmr"].fillna(pair_fill)
        df["avg_mmr"] = df["avg_mmr"].fillna(df["region"].map(stats["region_means"]))
        df["avg_mmr"] = df["avg_mmr"].fillna(stats["global_mean"])
    else:
        df["avg_mmr"] = df["avg_mmr"].fillna(stats["global_mean"])

    return df


def transform_mmr_value(df, mode):
    if "avg_mmr" not in df.columns:
        return df

    values = df["avg_mmr"].astype(float).clip(lower=0)

    if mode is None or mode == "raw":
        df["avg_mmr"] = values
    elif mode == "log1p":
        df["avg_mmr"] = np.log1p(values)
    elif mode == "sqrt":
        df["avg_mmr"] = np.sqrt(values)
    else:
        raise ValueError(f"Unknown mmr transform: {mode}")

    return df


def preprocess_fold(train_df, other_df, mmr_transform="raw"):
    train_df = to_pandas_df(train_df)
    other_df = to_pandas_df(other_df)

    train_df = extract_date_features(train_df)
    other_df = extract_date_features(other_df)

    if {"avg_mmr", "region", "game_mode"}.issubset(train_df.columns):
        mmr_stats = fit_mmr_fill_stats(train_df)
        train_df = transform_mmr_fill(train_df, mmr_stats)
        other_df = transform_mmr_fill(other_df, mmr_stats)

    train_df = transform_mmr_value(train_df, mmr_transform)
    other_df = transform_mmr_value(other_df, mmr_transform)

    return train_df, other_df


def make_preprocessor(target_cols, ohe_cols, numeric_cols):
    transformers = []

    if target_cols:
        transformers.append(
            (
                "target_enc",
                ce.TargetEncoder(
                    cols=target_cols,
                    handle_unknown="value",
                    handle_missing="value",
                    return_df=False,
                ),
                target_cols,
            )
        )

    if ohe_cols:
        transformers.append(
            (
                "ohe",
                ce.OneHotEncoder(
                    cols=ohe_cols,
                    handle_unknown="indicator",
                    handle_missing="indicator",
                    use_cat_names=True,
                    return_df=False,
                ),
                ohe_cols,
            )
        )

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_pipeline(target_cols, ohe_cols, numeric_cols):
    return Pipeline([
        ("prep", make_preprocessor(target_cols, ohe_cols, numeric_cols)),
        ("clf", cuLogisticRegression(max_iter=1000, C=1.0)),
    ])


def get_positive_class_proba(model, X):
    proba = model.predict_proba(X)

    if isinstance(proba, pd.DataFrame):
        if proba.shape[1] == 1:
            return proba.iloc[:, 0].to_numpy(dtype=float)
        return proba.iloc[:, 1].to_numpy(dtype=float)

    if isinstance(proba, pd.Series):
        return proba.to_numpy(dtype=float)

    arr = to_numpy(proba)

    if arr.ndim == 1:
        return arr.astype(float)

    return arr[:, 1].astype(float)


def prepare_model_data(df, feature_cols, target_col="radiant_win"):
    X = df.reindex(columns=feature_cols)

    if target_col in df.columns:
        y = df[target_col].astype(int).copy()
        return X, y

    return X, None


def cross_validate_model(
    df_train,
    target_cols,
    ohe_cols,
    numeric_cols,
    mmr_transform="raw",
    n_splits=5,
    target_col="radiant_win",
):
    df_train = to_pandas_df(df_train)
    df_train["date"] = pd.to_datetime(df_train["date"])
    df_train = df_train.sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_ginis = []

    feature_cols = target_cols + ohe_cols + numeric_cols
    feature_cols = list(dict.fromkeys(feature_cols))

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(df_train), start=1):
        train_fold = df_train.iloc[train_idx]
        valid_fold = df_train.iloc[valid_idx]

        train_fold, valid_fold = preprocess_fold(
            train_fold,
            valid_fold,
            mmr_transform=mmr_transform,
        )

        X_train, y_train = prepare_model_data(
            train_fold,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        X_valid, y_valid = prepare_model_data(
            valid_fold,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        model = make_pipeline(
            target_cols=target_cols,
            ohe_cols=ohe_cols,
            numeric_cols=numeric_cols,
        )
        model.fit(X_train, y_train)

        valid_pred = get_positive_class_proba(model, X_valid)

        fold_gini = float(gini_score(y_valid, valid_pred))
        fold_ginis.append(fold_gini)

        print(f"  fold {fold}: Gini = {fold_gini:.5f}")

    mean_gini = float(np.mean(fold_ginis))
    print(f"  mean Gini = {mean_gini:.5f}")

    return mean_gini, fold_ginis


def fit_full_model_and_predict(
    df_train,
    df_test,
    target_cols,
    ohe_cols,
    numeric_cols,
    mmr_transform="raw",
    target_col="radiant_win",
):
    df_train = to_pandas_df(df_train)
    df_test = to_pandas_df(df_test)

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_test["date"] = pd.to_datetime(df_test["date"])

    df_train = df_train.sort_values("date").reset_index(drop=True)

    df_train_prep, df_test_prep = preprocess_fold(
        df_train,
        df_test,
        mmr_transform=mmr_transform,
    )

    feature_cols = target_cols + ohe_cols + numeric_cols
    feature_cols = list(dict.fromkeys(feature_cols))

    X_train, y_train = prepare_model_data(
        df_train_prep,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    X_test, _ = prepare_model_data(
        df_test_prep,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    model = make_pipeline(
        target_cols=target_cols,
        ohe_cols=ohe_cols,
        numeric_cols=numeric_cols,
    )
    model.fit(X_train, y_train)

    test_pred = get_positive_class_proba(model, X_test)

    return model, test_pred


def save_test_predictions(df_test, test_pred, path="submission.csv", id_col="match_id"):
    df_test = to_pandas_df(df_test)
    test_pred = to_numpy(test_pred).reshape(-1).astype(float)

    if len(df_test) != len(test_pred):
        raise ValueError(
            f"Length mismatch: len(df_test)={len(df_test)} != len(test_pred)={len(test_pred)}"
        )

    if id_col in df_test.columns:
        ids = df_test[id_col].to_numpy()
    else:
        ids = df_test.index.to_numpy()

    submission = pd.DataFrame({
        "ID": ids,
        "Value": test_pred,
    })

    submission.to_csv(path, index=False)
    print(f"Saved submission: {path}")

    return submission


def run(
    df_train,
    df_test,
    n_splits=5,
    results_path="results.json",
    submission_path="submission.csv",
    submission_id_col="match_id",
    target_col="radiant_win",
):
    global results_store
    results_store = []

    date_cols = ["month", "is_weekend"]

    configs = {
        "dates": {
            "target_cols": [],
            "ohe_cols": date_cols,
            "numeric_cols": [],
            "mmr_transform": "raw",
        },
        "regions target encoding": {
            "target_cols": ["region"],
            "ohe_cols": [],
            "numeric_cols": [],
            "mmr_transform": "raw",
        },
        "regions OHE": {
            "target_cols": [],
            "ohe_cols": ["region"],
            "numeric_cols": [],
            "mmr_transform": "raw",
        },
        "dates+regions": {
            "target_cols": [],
            "ohe_cols":  ["region"] + date_cols,
            "numeric_cols": [],
            "mmr_transform": "raw",
        },
        "all (raw mmr)": {
            "target_cols": [],
            "ohe_cols": ["game_mode", "region"] + date_cols,
            "numeric_cols": ["avg_mmr", "mmr_missing", "duration"],
            "mmr_transform": "raw",
        },
        "all (sqrt mmr)": {
            "target_cols": [],
            "ohe_cols": ["game_mode", "region"] + date_cols,
            "numeric_cols": ["avg_mmr", "mmr_missing", "duration"],
            "mmr_transform": "sqrt",
        },
    }

    best_name = None
    best_cfg = None
    best_gini = -np.inf
    best_fold_ginis = None

    for config_name, cfg in configs.items():
        print(f"\n{config_name}")

        mean_gini, fold_ginis = cross_validate_model(
            df_train=df_train,
            target_cols=cfg["target_cols"],
            ohe_cols=cfg["ohe_cols"],
            numeric_cols=cfg["numeric_cols"],
            mmr_transform=cfg["mmr_transform"],
            n_splits=n_splits,
            target_col=target_col,
        )

        results_store.append({
            "timestamp": datetime.now().isoformat(),
            "model": "cuML LogisticRegression",
            "features": config_name,
            "target_cols": cfg["target_cols"],
            "ohe_cols": cfg["ohe_cols"],
            "numeric_cols": cfg["numeric_cols"],
            "mmr_transform": cfg["mmr_transform"],
            "fold_ginis": [round(x, 6) for x in fold_ginis],
            "gini": round(mean_gini, 6),
        })

        if mean_gini > best_gini:
            best_gini = mean_gini
            best_name = config_name
            best_cfg = cfg.copy()
            best_fold_ginis = fold_ginis

    print(f"\nBest config: {best_name}")
    print(f"Best mean Gini: {best_gini:.5f}")

    final_model, test_pred = fit_full_model_and_predict(
        df_train=df_train,
        df_test=df_test,
        target_cols=best_cfg["target_cols"],
        ohe_cols=best_cfg["ohe_cols"],
        numeric_cols=best_cfg["numeric_cols"],
        mmr_transform=best_cfg["mmr_transform"],
        target_col=target_col,
    )

    submission = save_test_predictions(
        df_test=df_test,
        test_pred=test_pred,
        path=submission_path,
        id_col=submission_id_col,
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_store, f, indent=2, ensure_ascii=False)

    return {
        "best_model": final_model,
        "best_config": best_name,
        "best_params": best_cfg,
        "best_gini": best_gini,
        "best_fold_ginis": best_fold_ginis,
        "test_pred": test_pred,
        "submission": submission,
        "cv_results": pd.DataFrame(results_store),
    }