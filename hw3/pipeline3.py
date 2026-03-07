import numpy as np
from sklearn.metrics import roc_auc_score
import category_encoders as ce
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1.0

def align_to_train_columns(df, train_columns):
    df = df.copy()

    for col in train_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[train_columns]

def make_month_weekend_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df.drop(columns=["weekday"], inplace=True)
    return df

def fit_mmr_fill(df_train: pd.DataFrame):
    group_means = df_train.groupby(["region", "game_mode"])["avg_mmr"].mean()
    region_means = df_train.groupby("region")["avg_mmr"].mean()
    global_mean = df_train["avg_mmr"].mean()

    return {
        "group_means": group_means,
        "region_means": region_means,
        "global_mean": global_mean,
    }


def transform_mmr_fill(df, mmr_stats):
    df = df.copy()

    idx = pd.MultiIndex.from_frame(df[["region", "game_mode"]])
    df["avg_mmr"] = df["avg_mmr"].fillna(
        pd.Series(idx.map(mmr_stats["group_means"]), index=df.index)
    )
    df["avg_mmr"] = df["avg_mmr"].fillna(df["region"].map(mmr_stats["region_means"]))
    df["avg_mmr"] = df["avg_mmr"].fillna(mmr_stats["global_mean"])

    return df

def fit_encoder(df_train, region_features=True, date_features=True):
    cols = []

    if date_features:
        cols.append("month")

    if region_features:
        cols.extend(["region", "game_mode"])

    if not cols:
        return None

    ohe = ce.OneHotEncoder(
        cols=cols,
        use_cat_names=True,
        handle_unknown="indicator"
    )
    ohe.fit(df_train)
    return ohe


def transform_encoder(df, encoder):
    df = df.copy()
    if encoder is None:
        return df
    return encoder.transform(df)


def preprocess_fold(df_train: pd.DataFrame, df_other: pd.DataFrame, region_features=True, date_features=True):
    df_train = df_train.copy()
    df_other = df_other.copy()

    df_train = make_month_weekend_features(df_train)
    df_other = make_month_weekend_features(df_other)

    df_other = align_to_train_columns(df_other, df_train.columns)

    mmr_stats = fit_mmr_fill(df_train)
    df_train = transform_mmr_fill(df_train, mmr_stats)
    df_other = transform_mmr_fill(df_other, mmr_stats)
    # print(df_train.head())
    # print(f"Nans for train: {df_train.isna().sum().sum()}")
    # print(df_other.head())
    # print(f"Nans for other: {df_other.isna().sum().sum()}")

    encoder = fit_encoder(
        df_train,
        region_features=region_features,
        date_features=date_features
    )

    df_train = transform_encoder(df_train, encoder)
    df_other = transform_encoder(df_other, encoder)

    df_train = df_train.drop(columns=["date", "duration", "match_id"], errors="ignore")
    df_other = df_other.drop(columns=["date", "duration", "match_id", "radiant_win"], errors="ignore")
    # print("========= AFTER PREPROCESS (without scaler) ==========")
    # print(df_train.head())
    # print(f"Nans for train: {df_train.isna().sum().sum()}")
    # print(df_other.head())
    # print(f"Nans for other: {df_other.isna().sum().sum()}")
    return df_train, df_other

def get_model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        raise ValueError(
            f"{type(model).__name__} does not support predict_proba or decision_function"
        )

def inspect_pipeline_transform(fitted_model, X):
    Xt = fitted_model.named_steps["preprocessor"].transform(X)
    cols = fitted_model.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame(Xt, columns=cols, index=X.index)


def start_preds(df_train_raw: pd.DataFrame, model: Pipeline, n_splits=5, target_col="radiant_win",
                region_features=True, date_features=True):

    df_train_raw = df_train_raw.copy()
    df_train_raw["date"] = pd.to_datetime(df_train_raw["date"])
    df_train_raw = df_train_raw.sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_scores = []
    oof_pred = np.zeros(len(df_train_raw))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_train_raw), start=1):
        train_fold_raw = df_train_raw.iloc[train_idx].copy()
        val_fold_raw = df_train_raw.iloc[val_idx].copy()

        y_train_fold = train_fold_raw[target_col].copy()
        y_val_fold = val_fold_raw[target_col].copy()

        X_train_fold, X_val_fold = preprocess_fold(
            train_fold_raw,
            val_fold_raw,
            region_features=region_features,
            date_features=date_features
        )

        X_train_fold = X_train_fold.drop(columns=[target_col], errors="ignore")
        X_val_fold = X_val_fold.drop(columns=[target_col], errors="ignore")

        current_model: Pipeline = clone(model)
        current_model.fit(X_train_fold, y_train_fold)
        # debug_df = inspect_pipeline_transform(current_model, X_train_fold)
        # debug_df.to_csv("debug.csv")
        val_pred = get_model_scores(current_model, X_val_fold)

        fold_gini = gini(y_val_fold, val_pred)
        fold_scores.append(fold_gini)
        oof_pred[val_idx] = val_pred

        print(f"Fold {fold}: Gini = {fold_gini:.5f}")

    return fold_scores, oof_pred


def run_pipeline(df_train: pd.DataFrame, df_test: pd.DataFrame, model: Pipeline, n_splits=5,
                 region_features: bool = True, date_features: bool = True,
                 target_col: str = "radiant_win"):
    scores, oof_pred = start_preds(
        df_train_raw=df_train,
        model=model,
        n_splits=n_splits,
        target_col=target_col,
        region_features=region_features,
        date_features=date_features
    )

    print("Mean Gini:", np.mean(scores))

    df_train_sorted = df_train.copy()
    df_train_sorted["date"] = pd.to_datetime(df_train_sorted["date"])
    df_train_sorted = df_train_sorted.sort_values("date").reset_index(drop=True)

    y = df_train_sorted[target_col].copy()

    X_train_full, X_test = preprocess_fold(
        df_train_sorted,
        df_test,
        region_features=region_features,
        date_features=date_features
    )

    X_train_full = X_train_full.drop(columns=[target_col], errors="ignore")

    final_model = clone(model)
    final_model.fit(X_train_full, y)

    test_pred = get_model_scores(final_model, X_test)

    return scores, oof_pred, test_pred, final_model


