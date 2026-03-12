import json
import os
import re
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache

import cudf
import numpy as np
import cupy as cp
import pandas as pd
import category_encoders as ce
import scipy.sparse as sp
import cupyx.scipy.sparse as cusparse
from pymystem3 import Mystem
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from heroes_encoder import HeroesEncoder


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*No categorical columns found.*")


@dataclass(frozen=True)
class FeatureConfig:
    target_cols: tuple[str, ...] = ()
    ohe_cols: tuple[str, ...] = ()
    numeric_cols: tuple[str, ...] = ()
    text_cols: tuple[str, ...] = ()
    mmr_transform: str = "raw"

    text_max_features: int = 5000
    text_min_df: int | float = 10
    text_max_df: int | float = 0.8
    text_ngram_range: tuple[int, int] = (1, 2)

    @property
    def feature_cols(self):
        return list(dict.fromkeys((
            *self.target_cols,
            *self.ohe_cols,
            *self.numeric_cols,
        )))

    def with_mmr(self, transform="raw"):
        return replace(
            self,
            numeric_cols=tuple(dict.fromkeys((
                *self.numeric_cols,
                "avg_mmr",
                "mmr_missing",
            ))),
            mmr_transform=transform,
        )


def unique_list(values):
    return list(dict.fromkeys(values))


def to_pandas_df(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pd.Series):
        return df.to_frame().copy()
    if hasattr(df, "to_pandas"):
        out = df.to_pandas()
        if isinstance(out, pd.Series):
            return out.to_frame().copy()
        return out.copy()
    return pd.DataFrame(df).copy()


def to_gpu_dense(x, dtype=cp.float32):
    if x is None:
        return None
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    return cp.asarray(x, dtype=dtype)


def to_gpu_labels(y):
    if y is None:
        return None
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return cp.asarray(y, dtype=cp.int32)


def issparse_convert(X):
    if X is None:
        return None
    if cusparse.issparse(X):
        return cusparse.csr_matrix(X)
    if sp.issparse(X):
        return cusparse.csr_matrix(X)
    return X


def as_feature_block(x):
    if x is None:
        return None
    if cusparse.issparse(x):
        return x.astype(cp.float32)
    if sp.issparse(x):
        return cusparse.csr_matrix(x, dtype=cp.float32)
    return to_gpu_dense(x, dtype=cp.float32)


def combine_feature_blocks(*blocks):
    blocks = [as_feature_block(b) for b in blocks if b is not None]
    if not blocks:
        raise ValueError("Не передано ни одного блока фичей")

    if any(cusparse.issparse(b) for b in blocks):
        blocks = [
            b if cusparse.issparse(b) else cusparse.csr_matrix(b)
            for b in blocks
        ]
        return cusparse.hstack(blocks, format="csr", dtype=cp.float32)

    if len(blocks) == 1:
        return blocks[0]

    return cp.hstack(blocks).astype(cp.float32, copy=False)


def has_any_nonzero(x):
    if x is None:
        return False
    if cusparse.issparse(x):
        return x.nnz > 0
    return bool(cp.count_nonzero(x).item())


def make_submission_path(base_path, suffix):
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".csv"
    return f"{root}_{suffix}{ext}"


def gini_score(y_true, y_score):
    y_true = cp.asarray(y_true).reshape(-1)
    y_score = cp.asarray(y_score).reshape(-1)
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0

_MYSTEM = Mystem()
_TOKEN_RE = re.compile(r"\w+|[()?!]+", flags=re.UNICODE)
_REPEAT_RE = re.compile(r"([a-zа-я])\1{2,}", flags=re.IGNORECASE)
_ALPHA_RE = re.compile(r"[a-zа-я]+", flags=re.IGNORECASE)



@lru_cache(maxsize=200_000)
def preprocess_text(text: str) -> tuple[str, ...]:
    if text is None:
        return ()
    if isinstance(text, float) and np.isnan(text):
        return ()

    text = str(text)
    if text == "":
        return ()

    text = text.replace("|", " ")
    text = text.lower()
    text = _REPEAT_RE.sub(r"\1\1", text)

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return ()

    mystem = _MYSTEM
    lemmas = mystem.lemmatize(" ".join(tokens))
    lemmas = [x.strip() for x in lemmas if x.strip()]
    lemmas = [x for x in lemmas if _ALPHA_RE.fullmatch(x)]
    lemmas = [x for x in lemmas if len(x) >= 3]

    return tuple(lemmas)


def normalize_chat_df(chat_df, text_cols):
    chat_df = to_pandas_df(chat_df)
    if "match_id" not in chat_df.columns:
        raise ValueError("chat_df должен содержать колонку match_id")

    chat_df = normalize_match_id(chat_df)
    text_cols = list(text_cols or [])

    if text_cols:
        missing = [col for col in text_cols if col not in chat_df.columns]
        if missing:
            raise ValueError(f"В chat_df нет колонок: {missing}")

        chat_df = chat_df.copy()
        for col in text_cols:
            chat_df[col] = chat_df[col].fillna("")

    chat_df = chat_df.dropna(subset=["match_id"]).copy()
    chat_df = chat_df.drop_duplicates("match_id", keep="last")
    return chat_df


def _texts_from_matches(matches_df, text_col):
    return matches_df[text_col].fillna("").tolist()


def _texts_from_chat(chat_indexed, match_ids, text_col):
    match_ids = pd.Index(pd.to_numeric(pd.Series(match_ids), errors="coerce"))
    return chat_indexed[text_col].reindex(match_ids).fillna("").tolist()



def normalize_match_id(df):
    df = to_pandas_df(df)
    if "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    return df


def normalize_players_schema(df):
    df = to_pandas_df(df)
    for col in ("match_id", "hero_id", "player_slot"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_players_by_match_ids(players_df, match_ids):
    players_df = normalize_players_schema(players_df)
    match_ids = pd.Index(pd.Series(match_ids).dropna().unique())
    if "match_id" not in players_df.columns:
        return players_df.iloc[0:0].copy()
    return players_df[players_df["match_id"].isin(match_ids)].copy()


def preprocess_players_df(players_df):
    players_df = normalize_players_schema(players_df)

    if "account_id" in players_df.columns:
        players_df = players_df[players_df["account_id"] != -1].copy()

    required_cols = [col for col in ("match_id", "hero_id", "player_slot") if col in players_df.columns]
    if required_cols:
        players_df = players_df.dropna(subset=required_cols)

    if not {"match_id", "hero_id"}.issubset(players_df.columns):
        return players_df

    players_df["hero_is_zero"] = (players_df["hero_id"] == 0).astype(np.int8)

    agg = (
        players_df
        .groupby("match_id", dropna=False)
        .agg(
            rows_in_match=("match_id", "size"),
            unique_heroes=("hero_id", "nunique"),
            hero_zero_cnt=("hero_is_zero", "sum"),
        )
        .reset_index()
    )

    bad_match_ids = agg.loc[
        (agg["rows_in_match"] != 10) |
        (agg["unique_heroes"] != 10) |
        (agg["hero_zero_cnt"] > 0),
        "match_id",
    ]

    if len(bad_match_ids) > 0:
        players_df = players_df[~players_df["match_id"].isin(pd.Index(bad_match_ids))].copy()

    players_df = players_df.drop(columns=["hero_is_zero"], errors="ignore")
    return players_df.dropna(subset=["match_id", "hero_id", "player_slot"])


def split_players_by_matches(player_df, matches_train_df, matches_test_df):
    players_df = normalize_players_schema(player_df)
    train_matches_df = normalize_match_id(matches_train_df)
    test_matches_df = normalize_match_id(matches_test_df)

    train_ids = pd.Index(train_matches_df["match_id"].dropna().unique()) if "match_id" in train_matches_df.columns else pd.Index([])
    test_ids = pd.Index(test_matches_df["match_id"].dropna().unique()) if "match_id" in test_matches_df.columns else pd.Index([])

    overlap = train_ids.intersection(test_ids)
    if len(overlap) > 0:
        raise ValueError("Один и тот же match_id попал и в train, и в test")

    players_train_df = players_df[players_df["match_id"].isin(train_ids)].copy()
    players_test_df = players_df[players_df["match_id"].isin(test_ids)].copy()

    return players_train_df, players_test_df

def extract_date_features(df):
    df = to_pandas_df(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month.astype(str)
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(str)
    return df


def fit_mmr_fill_stats(df):
    df = to_pandas_df(df)

    group_means = (
        df.groupby(["region", "game_mode"], dropna=False, as_index=False)["avg_mmr"]
        .mean()
        .rename(columns={"avg_mmr": "group_avg_mmr"})
    )
    region_means = (
        df.groupby("region", dropna=False, as_index=False)["avg_mmr"]
        .mean()
        .rename(columns={"avg_mmr": "region_avg_mmr"})
    )
    global_mean = float(pd.to_numeric(df["avg_mmr"], errors="coerce").mean())
    if not np.isfinite(global_mean):
        global_mean = 0.0

    return {
        "group_means": group_means,
        "region_means": region_means,
        "global_mean": global_mean,
    }


def transform_mmr_fill(df, stats):
    if "avg_mmr" not in df.columns:
        return df

    df = to_pandas_df(df)
    df["avg_mmr"] = pd.to_numeric(df["avg_mmr"], errors="coerce")
    df["mmr_missing"] = df["avg_mmr"].isna().astype(np.int8)

    if {"region", "game_mode"}.issubset(df.columns):
        df = df.merge(stats["group_means"], on=["region", "game_mode"], how="left")
        df = df.merge(stats["region_means"], on="region", how="left")

        df["avg_mmr"] = df["avg_mmr"].fillna(df["group_avg_mmr"])
        df["avg_mmr"] = df["avg_mmr"].fillna(df["region_avg_mmr"])
        df["avg_mmr"] = df["avg_mmr"].fillna(stats["global_mean"])

        df = df.drop(columns=["group_avg_mmr", "region_avg_mmr"], errors="ignore")
    else:
        df["avg_mmr"] = df["avg_mmr"].fillna(stats["global_mean"])

    return df


def transform_mmr_value(df, mode):
    if "avg_mmr" not in df.columns:
        return df

    df = to_pandas_df(df)
    values = pd.to_numeric(df["avg_mmr"], errors="coerce").fillna(0).astype(np.float32).clip(lower=0)

    match mode:
        case "raw":
            df["avg_mmr"] = values
        case "log1p":
            df["avg_mmr"] = np.log1p(values)
        case "sqrt":
            df["avg_mmr"] = np.sqrt(values)
        case _:
            df["avg_mmr"] = values

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



def to_numpy_2d(X):
    if sp.issparse(X):
        return X
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X

def ensure_float32_block(X):
    if sp.issparse(X):
        return X.astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def make_preprocessor(target_cols, ohe_cols, numeric_cols):
    transformers = []

    if target_cols:
        transformers.append(
            (
                "target_enc",
                Pipeline([
                    ("enc", ce.TargetEncoder(
                        cols=list(target_cols),
                        handle_unknown="value",
                        handle_missing="value",
                        return_df=False,
                    )),
                    ("ensure_2d", FunctionTransformer(to_numpy_2d, validate=False)),
                ]),
                list(target_cols),
            )
        )

    if ohe_cols:
        transformers.append(
            (
                "ohe",
                Pipeline([
                    ("to_numpy", FunctionTransformer(to_numpy_2d, validate=False)),
                    ("enc", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float32,
                    )),
                ]),
                list(ohe_cols),
            )
        )

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("to_numpy", FunctionTransformer(to_numpy_2d, validate=False)),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                list(numeric_cols),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def prepare_model_data(df, feature_cols, target_col="radiant_win"):
    X = df.reindex(columns=feature_cols)

    if target_col in df.columns:
        y = df[target_col].astype(np.int32).copy()
        return X, y

    return X, None


def fit_tabular_feature_blocks(train_matches, other_matches, cfg: FeatureConfig, target_col="radiant_win"):
    train_prep, other_prep = preprocess_fold(
        train_matches,
        other_matches,
        mmr_transform=cfg.mmr_transform,
    )

    X_train_tab, y_train = prepare_model_data(
        train_prep,
        feature_cols=cfg.feature_cols,
        target_col=target_col,
    )
    X_other_tab, y_other = prepare_model_data(
        other_prep,
        feature_cols=cfg.feature_cols,
        target_col=target_col,
    )

    prep = make_preprocessor(
        target_cols=list(cfg.target_cols),
        ohe_cols=list(cfg.ohe_cols),
        numeric_cols=list(cfg.numeric_cols),
    )

    X_train_block = prep.fit_transform(X_train_tab, y_train)
    X_other_block = prep.transform(X_other_tab)

    if sp.issparse(X_train_block):
        X_train_block = X_train_block.astype(np.float32)
    else:
        X_train_block = np.asarray(X_train_block, dtype=np.float32)

    if sp.issparse(X_other_block):
        X_other_block = X_other_block.astype(np.float32)
    else:
        X_other_block = np.asarray(X_other_block, dtype=np.float32)

    return prep, X_train_block, to_gpu_labels(y_train), X_other_block, to_gpu_labels(y_other)


def fit_text_feature_blocks(train_matches, other_matches, chat_df, cfg: FeatureConfig):
    if not cfg.text_cols:
        return [], []

    if chat_df is None:
        missing = [
            col for col in cfg.text_cols
            if col not in train_matches.columns or col not in other_matches.columns
        ]
        if missing:
            raise ValueError(f"В matches нет текстовых колонок: {missing}")
        chat_indexed = None
    else:
        chat_df = normalize_chat_df(chat_df, cfg.text_cols)
        chat_indexed = chat_df.set_index("match_id")
        if not chat_indexed.index.is_unique:
            chat_indexed = chat_indexed[~chat_indexed.index.duplicated(keep="last")]

    train_blocks = []
    other_blocks = []

    for text_col in cfg.text_cols:
        if chat_indexed is None:
            train_raw = _texts_from_matches(train_matches, text_col)
            other_raw = _texts_from_matches(other_matches, text_col)
        else:
            train_raw = _texts_from_chat(chat_indexed, train_matches["match_id"], text_col)
            other_raw = _texts_from_chat(chat_indexed, other_matches["match_id"], text_col)

        train_clean = [" ".join(preprocess_text(x)) for x in train_raw]
        other_clean = [" ".join(preprocess_text(x)) for x in other_raw]

        vectorizer = TfidfVectorizer(
            max_features=cfg.text_max_features,
            min_df=cfg.text_min_df,
            max_df=cfg.text_max_df,
            ngram_range=cfg.text_ngram_range,
        )

        train_clean = cudf.Series(train_clean)
        other_clean = cudf.Series(other_clean)
        X_train_text = vectorizer.fit_transform(train_clean)
        X_other_text = vectorizer.transform(other_clean)

        train_blocks.append(X_train_text)
        other_blocks.append(X_other_text)

    return train_blocks, other_blocks


def make_logreg(model_params=None):
    model_params = model_params or {}

    return cuLogisticRegression(
        penalty="l2",
        C=model_params.get("C", 1.0),
        max_iter=model_params.get("max_iter", 2000),
    )


def get_positive_class_proba(model, X):
    proba = model.predict_proba(X)

    if isinstance(proba, pd.DataFrame):
        proba = proba.to_numpy()
    elif isinstance(proba, pd.Series):
        proba = proba.to_numpy()

    arr = cp.asarray(proba)

    if arr.ndim == 1:
        return arr.astype(cp.float32, copy=False)

    if arr.shape[1] == 1:
        return arr[:, 0].astype(cp.float32, copy=False)

    return arr[:, 1].astype(cp.float32, copy=False)


def build_feature_matrices(
    train_matches,
    other_matches,
    cfg: FeatureConfig,
    *,
    train_players=None,
    other_players=None,
    chat_df=None,
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    target_col="radiant_win",
):
    use_text = bool(cfg.text_cols)

    if not use_tabular and not use_heroes and not use_text:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    train_matches = to_pandas_df(train_matches)
    other_matches = to_pandas_df(other_matches)

    X_train_blocks = []
    X_other_blocks = []

    y_train = (
        to_gpu_labels(train_matches[target_col].astype(np.int32))
        if target_col in train_matches.columns
        else None
    )
    y_other = (
        to_gpu_labels(other_matches[target_col].astype(np.int32))
        if target_col in other_matches.columns
        else None
    )

    if use_tabular:
        _, X_train_tab, y_train_tab, X_other_tab, y_other_tab = fit_tabular_feature_blocks(
            train_matches,
            other_matches,
            cfg,
            target_col=target_col,
        )
        X_train_blocks.append(X_train_tab)
        X_other_blocks.append(X_other_tab)

        if y_train_tab is not None:
            y_train = y_train_tab
        if y_other_tab is not None:
            y_other = y_other_tab

    if use_text:
        X_train_text_blocks, X_other_text_blocks = fit_text_feature_blocks(
            train_matches,
            other_matches,
            chat_df,
            cfg,
        )
        X_train_blocks.extend(X_train_text_blocks)
        X_other_blocks.extend(X_other_text_blocks)

    if use_heroes:
        hero_encoder = HeroesEncoder(
            heroes_df=heroes_df,
            split_teams=split_teams,
            dtype=np.int8,
        )

        X_train_heroes = hero_encoder.fit_transform(
            train_players,
            matches_df=train_matches[["match_id"]],
        )
        X_other_heroes = hero_encoder.transform(
            other_players,
            matches_df=other_matches[["match_id"]],
        )

        X_train_blocks.append(X_train_heroes)
        X_other_blocks.append(X_other_heroes)


    X_train = combine_feature_blocks(*X_train_blocks)
    X_other = combine_feature_blocks(*X_other_blocks)

    if X_train.shape[1] == 0:
        raise ValueError("После сборки матрицы признаков не осталось ни одной фичи")

    return X_train, y_train, X_other, y_other


def cross_validate(
    df_train,
    cfg: FeatureConfig,
    *,
    players_train=None,
    chat_df=None,
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    n_splits=5,
    target_col="radiant_win",
    model_params=None,
):
    from sklearn.model_selection import TimeSeriesSplit

    if not use_tabular and not use_heroes and not cfg.text_cols:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    df_train = to_pandas_df(df_train)
    if use_heroes:
        players_train = to_pandas_df(players_train)

    df_train["date"] = pd.to_datetime(df_train["date"], errors="coerce")
    df_train = df_train.sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_ginis = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(df_train), start=1):
        train_matches = df_train.iloc[train_idx].copy()
        valid_matches = df_train.iloc[valid_idx].copy()

        train_players_fold = None
        valid_players_fold = None

        if use_heroes:
            train_players_fold = filter_players_by_match_ids(
                players_train,
                train_matches["match_id"],
            )
            valid_players_fold = filter_players_by_match_ids(
                players_train,
                valid_matches["match_id"],
            )


        X_train, y_train, X_valid, y_valid = build_feature_matrices(
            train_matches,
            valid_matches,
            cfg,
            train_players=train_players_fold,
            other_players=valid_players_fold,
            chat_df=chat_df,
            use_tabular=use_tabular,
            use_heroes=use_heroes,
            heroes_df=heroes_df,
            split_teams=split_teams,
            target_col=target_col,
        )

        model = make_logreg(model_params=model_params)

        X_train = issparse_convert(X_train)
        X_valid = issparse_convert(X_valid)

        model.fit(X_train, y_train)

        valid_pred = get_positive_class_proba(model, X_valid)
        fold_gini = float(gini_score(y_valid, valid_pred))
        fold_ginis.append(fold_gini)

        print(f"  fold {fold}: Gini = {fold_gini:.5f}")

    mean_gini = float(np.mean(fold_ginis))
    print(f"  mean Gini = {mean_gini:.5f}")

    return mean_gini, fold_ginis


def fit_model_and_predict(
    df_train,
    df_test,
    cfg: FeatureConfig,
    *,
    players_train=None,
    players_test=None,
    chat_df=None,
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    target_col="radiant_win",
    model_params=None,
):
    if not use_tabular and not use_heroes and not cfg.text_cols:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    df_train = to_pandas_df(df_train)
    df_test = to_pandas_df(df_test)

    if use_heroes:
        players_train = to_pandas_df(players_train)
        players_test = to_pandas_df(players_test)

    df_train["date"] = pd.to_datetime(df_train["date"], errors="coerce")
    df_test["date"] = pd.to_datetime(df_test["date"], errors="coerce")
    df_train = df_train.sort_values("date").reset_index(drop=True)

    if use_heroes and len(players_test) == 0:
        warnings.warn("players_test пустой")

    X_train, y_train, X_test, _ = build_feature_matrices(
        df_train,
        df_test,
        cfg,
        train_players=players_train,
        other_players=players_test,
        chat_df=chat_df,
        use_tabular=use_tabular,
        use_heroes=use_heroes,
        heroes_df=heroes_df,
        split_teams=split_teams,
        target_col=target_col,
    )

    if use_heroes and not has_any_nonzero(X_test):
        warnings.warn("Фича героя для теста пустая")

    model = make_logreg(model_params=model_params)

    X_train = issparse_convert(X_train)
    X_test = issparse_convert(X_test)

    model.fit(X_train, y_train)
    test_pred = get_positive_class_proba(model, X_test)
    return model, test_pred


def save_test_predictions(df_test, test_pred, path="submission.csv", id_col="match_id"):
    df_test = to_pandas_df(df_test)

    if isinstance(test_pred, cp.ndarray):
        test_pred = cp.asnumpy(test_pred)
    else:
        test_pred = np.asarray(test_pred)

    test_pred = test_pred.reshape(-1).astype(float)

    if len(df_test) != len(test_pred):
        raise ValueError(
            f"Length mismatch: len(df_test)={len(df_test)} != len(test_pred)={len(test_pred)}"
        )

    if id_col in df_test.columns:
        ids = df_test[id_col]
    else:
        ids = df_test.index

    submission = pd.DataFrame({
        "ID": ids,
        "Value": test_pred,
    })

    submission.to_csv(path, index=False)
    print(f"Saved submission: {path}")

    return submission


def prepare_players_for_run(players_train, players_test, df_train, df_test, preprocess_players=True):
    players_train = normalize_players_schema(players_train)
    players_test = normalize_players_schema(players_test)

    if not preprocess_players:
        return players_train, players_test

    base_cols = list(dict.fromkeys(
        list(players_train.columns) + list(players_test.columns)
    ))

    if "match_id" not in base_cols:
        raise KeyError(f"players_train/players_test have no 'match_id'. columns={base_cols}")

    players_all = pd.concat(
        [
            players_train.reindex(columns=base_cols),
            players_test.reindex(columns=base_cols),
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    players_all = preprocess_players_df(players_all)

    if "match_id" not in players_all.columns:
        raise KeyError(
            f"players_all lost 'match_id' after preprocess_players_df. columns={list(players_all.columns)}"
        )

    train_ids = pd.Index(to_pandas_df(df_train)["match_id"].dropna().unique())
    test_ids = pd.Index(to_pandas_df(df_test)["match_id"].dropna().unique())

    players_train = players_all.loc[players_all["match_id"].isin(train_ids)].copy()
    players_test = players_all.loc[players_all["match_id"].isin(test_ids)].copy()

    return players_train, players_test

def log_result(results_store, stage, config_name, cfg: FeatureConfig, mean_gini, fold_ginis, model_name="cuML LogisticRegression"):
    results_store.append({
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "model": model_name,
        "features": config_name,
        "target_cols": list(cfg.target_cols),
        "ohe_cols": list(cfg.ohe_cols),
        "numeric_cols": list(cfg.numeric_cols),
        "text_cols": list(cfg.text_cols),
        "mmr_transform": cfg.mmr_transform,
        "text_max_features": cfg.text_max_features if cfg.text_cols else None,
        "text_min_df": cfg.text_min_df if cfg.text_cols else None,
        "text_max_df": cfg.text_max_df if cfg.text_cols else None,
        "text_ngram_range": cfg.text_ngram_range if cfg.text_cols else None,
        "fold_ginis": [round(float(x), 6) for x in fold_ginis],
        "gini": round(float(mean_gini), 6),
    })


def build_base_configs():
    date_cols = ("month", "is_weekend")
    return {
        "dates": FeatureConfig(
            ohe_cols=date_cols,
        ),
        "regions": FeatureConfig(
            ohe_cols=("region",),
        ),
        "dates+regions (OHE)": FeatureConfig(
            ohe_cols=("region",) + date_cols,
        ),
        "all base (no mmr)": FeatureConfig(
            ohe_cols=("game_mode", "region") + date_cols,
        ),
    }


# def stage_base_search(df_train, n_splits, target_col, results_store, base_configs=None):
#     base_configs = build_base_configs() if base_configs is None else base_configs
#
#     best_base_name = None
#     best_base_cfg = None
#     best_base_gini = -np.inf
#     best_base_fold_ginis = None
#
#     print("\nStage 1: search best base config without avg_mmr")
#
#     for config_name, cfg in base_configs.items():
#         print(f"\n{config_name}")
#
#         mean_gini, fold_ginis = cross_validate(
#             df_train=df_train,
#             cfg=cfg,
#             use_tabular=True,
#             use_heroes=False,
#             n_splits=n_splits,
#             target_col=target_col,
#         )
#
#         log_result(
#             results_store=results_store,
#             stage="base_search",
#             config_name=config_name,
#             cfg=cfg,
#             mean_gini=mean_gini,
#             fold_ginis=fold_ginis,
#             model_name="cuML LogisticRegression",
#         )
#
#         if mean_gini > best_base_gini:
#             best_base_gini = mean_gini
#             best_base_name = config_name
#             best_base_cfg = cfg
#             best_base_fold_ginis = fold_ginis
#
#     print(f"\nBest base config: {best_base_name}")
#     print(f"Best base mean Gini: {best_base_gini:.5f}")
#
#     return {
#         "best_base_name": best_base_name,
#         "best_base_cfg": best_base_cfg,
#         "best_base_gini": best_base_gini,
#         "best_base_fold_ginis": best_base_fold_ginis,
#     }


# def stage_mmr_comparison(
#     df_train,
#     best_base_cfg,
#     transformed_mmr,
#     n_splits,
#     target_col,
#     results_store,
# ):
#     comparison_configs = {
#         f"base + {transformed_mmr} avg_mmr + mmr_missing": best_base_cfg.with_mmr(transformed_mmr),
#     }
#
#     comparison_scores = {}
#     print("\nStage 2: compare base vs new MMR models")
#
#     for config_name, cfg in comparison_configs.items():
#         print(f"\n{config_name}")
#
#         mean_gini, fold_ginis = cross_validate(
#             df_train=df_train,
#             cfg=cfg,
#             use_tabular=True,
#             use_heroes=False,
#             n_splits=n_splits,
#             target_col=target_col,
#         )
#
#         comparison_scores[config_name] = {
#             "mean_gini": mean_gini,
#             "fold_ginis": fold_ginis,
#             "cfg": cfg,
#         }
#
#         log_result(
#             results_store=results_store,
#             stage="mmr_comparison",
#             config_name=config_name,
#             cfg=cfg,
#             mean_gini=mean_gini,
#             fold_ginis=fold_ginis,
#             model_name="cuML LogisticRegression",
#         )
#
#     best_tabular_name = max(
#         comparison_scores,
#         key=lambda name: comparison_scores[name]["mean_gini"],
#     )
#     best_tabular_cfg = comparison_scores[best_tabular_name]["cfg"]
#
#     print(f"\nBest tabular config after MMR comparison: {best_tabular_name}")
#     print(f"Best tabular mean Gini: {comparison_scores[best_tabular_name]['mean_gini']:.5f}")
#
#     return {
#         "best_tabular_name": best_tabular_name,
#         "best_tabular_cfg": best_tabular_cfg,
#         "comparison_scores": comparison_scores,
#     }


# def stage_hero_models(
#     df_train,
#     players_train,
#     heroes_df,
#     split_teams,
#     n_splits,
#     target_col,
#     best_tabular_cfg,
#     results_store,
# ):
#     hero_only_cfg = FeatureConfig()
#
#     hero_model_specs = {
#         "all features": {
#             "cfg": best_tabular_cfg,
#             "use_tabular": True,
#             "use_heroes": True,
#         },
#         "heroes only": {
#             "cfg": hero_only_cfg,
#             "use_tabular": False,
#             "use_heroes": True,
#         },
#     }
#
#     hero_model_scores = {}
#     print("\nStage 3: compare hero-based models")
#
#     for config_name, spec in hero_model_specs.items():
#         print(f"\n{config_name}")
#
#         mean_gini, fold_ginis = cross_validate(
#             df_train=df_train,
#             cfg=spec["cfg"],
#             players_train=players_train,
#             use_tabular=spec["use_tabular"],
#             use_heroes=spec["use_heroes"],
#             heroes_df=heroes_df,
#             split_teams=split_teams,
#             n_splits=n_splits,
#             target_col=target_col,
#         )
#
#         hero_model_scores[config_name] = {
#             "mean_gini": mean_gini,
#             "fold_ginis": fold_ginis,
#             "cfg": spec["cfg"],
#         }
#
#         log_result(
#             results_store=results_store,
#             stage="hero_models",
#             config_name=config_name,
#             cfg=spec["cfg"],
#             mean_gini=mean_gini,
#             fold_ginis=fold_ginis,
#             model_name="cuml logistic regression with heroes",
#         )
#
#     print("\nHero models comparison (CV)")
#     print(f"All features Gini: {hero_model_scores['all features']['mean_gini']:.5f}")
#     print(f"Heroes only Gini: {hero_model_scores['heroes only']['mean_gini']:.5f}")
#
#     return hero_only_cfg, hero_model_scores


# def stage_fit_hero_models_and_save(
#     df_train,
#     df_test,
#     players_train,
#     players_test,
#     best_tabular_cfg,
#     hero_only_cfg,
#     heroes_df,
#     split_teams,
#     target_col,
#     submission_path,
#     submission_id_col,
# ):
#     print("\nStage 4: fit full hero models and save test submissions")
#
#     all_features_model, all_features_test_pred = fit_model_and_predict(
#         df_train=df_train,
#         df_test=df_test,
#         players_train=players_train,
#         players_test=players_test,
#         cfg=best_tabular_cfg,
#         use_tabular=True,
#         use_heroes=True,
#         heroes_df=heroes_df,
#         split_teams=split_teams,
#         target_col=target_col,
#     )
#
#     heroes_only_model, heroes_only_test_pred = fit_model_and_predict(
#         df_train=df_train,
#         df_test=df_test,
#         players_train=players_train,
#         players_test=players_test,
#         cfg=hero_only_cfg,
#         use_tabular=False,
#         use_heroes=True,
#         heroes_df=heroes_df,
#         split_teams=split_teams,
#         target_col=target_col,
#     )
#
#     all_features_submission_path = make_submission_path(submission_path, "all_features")
#     heroes_only_submission_path = make_submission_path(submission_path, "heroes_only")
#
#     all_features_submission = save_test_predictions(
#         df_test=df_test,
#         test_pred=all_features_test_pred,
#         path=all_features_submission_path,
#         id_col=submission_id_col,
#     )
#
#     heroes_only_submission = save_test_predictions(
#         df_test=df_test,
#         test_pred=heroes_only_test_pred,
#         path=heroes_only_submission_path,
#         id_col=submission_id_col,
#     )
#
#     return {
#         "all_features_model": all_features_model,
#         "all_features_test_pred": all_features_test_pred,
#         "all_features_submission": all_features_submission,
#         "all_features_submission_path": all_features_submission_path,
#         "heroes_only_model": heroes_only_model,
#         "heroes_only_test_pred": heroes_only_test_pred,
#         "heroes_only_submission": heroes_only_submission,
#         "heroes_only_submission_path": heroes_only_submission_path,
#     }


# def stage_finalize_results(results_store, results_path):
#     with open(results_path, "w", encoding="utf-8") as f:
#         json.dump(results_store, f, indent=2, ensure_ascii=False)
#
#     cv_results = pd.DataFrame(results_store)
#     base_search_results = cv_results[cv_results["stage"] == "base_search"].reset_index(drop=True)
#     mmr_comparison_results = cv_results[cv_results["stage"] == "mmr_comparison"].reset_index(drop=True)
#     hero_model_results = cv_results[cv_results["stage"] == "hero_models"].reset_index(drop=True)
#
#     return {
#         "cv_results": cv_results,
#         "base_search_results": base_search_results,
#         "mmr_comparison_results": mmr_comparison_results,
#         "hero_model_results": hero_model_results,
#     }


# def run(
#     df_train,
#     df_test,
#     players_df,
#     heroes_df=None,
#     split_teams=True,
#     n_splits=5,
#     results_path="results.json",
#     submission_path="submission.csv",
#     submission_id_col="match_id",
#     target_col="radiant_win",
#     transformed_mmr="sqrt",
#     preprocess_players=True,
# ):
#     results_store = []
#
#     players_all = to_polars_df(players_df)
#     if preprocess_players:
#         players_all = preprocess_players_df(players_all)
#
#     players_train, players_test = split_players_by_matches(players_all, df_train, df_test)
#
#     base_results = stage_base_search(
#         df_train=df_train,
#         n_splits=n_splits,
#         target_col=target_col,
#         results_store=results_store,
#     )
#
#     mmr_results = stage_mmr_comparison(
#         df_train=df_train,
#         best_base_cfg=base_results["best_base_cfg"],
#         transformed_mmr=transformed_mmr,
#         n_splits=n_splits,
#         target_col=target_col,
#         results_store=results_store,
#     )
#
#     hero_only_cfg, hero_model_scores = stage_hero_models(
#         df_train=df_train,
#         players_train=players_train,
#         heroes_df=heroes_df,
#         split_teams=split_teams,
#         n_splits=n_splits,
#         target_col=target_col,
#         best_tabular_cfg=mmr_results["best_tabular_cfg"],
#         results_store=results_store,
#     )
#
#     fit_results = stage_fit_hero_models_and_save(
#         df_train=df_train,
#         df_test=df_test,
#         players_train=players_train,
#         players_test=players_test,
#         best_tabular_cfg=mmr_results["best_tabular_cfg"],
#         hero_only_cfg=hero_only_cfg,
#         heroes_df=heroes_df,
#         split_teams=split_teams,
#         target_col=target_col,
#         submission_path=submission_path,
#         submission_id_col=submission_id_col,
#     )
#
#     finalize_results = stage_finalize_results(
#         results_store=results_store,
#         results_path=results_path,
#     )
#
#     return {
#         "best_base_config": base_results["best_base_name"],
#         "best_base_params": base_results["best_base_cfg"],
#         "best_base_gini": base_results["best_base_gini"],
#         "best_base_fold_ginis": base_results["best_base_fold_ginis"],
#         "best_tabular_config_name": mmr_results["best_tabular_name"],
#         "best_tabular_params": mmr_results["best_tabular_cfg"],
#         "hero_model_scores": hero_model_scores,
#         "all_features_model": fit_results["all_features_model"],
#         "all_features_test_pred": fit_results["all_features_test_pred"],
#         "all_features_submission": fit_results["all_features_submission"],
#         "all_features_submission_path": fit_results["all_features_submission_path"],
#         "heroes_only_model": fit_results["heroes_only_model"],
#         "heroes_only_test_pred": fit_results["heroes_only_test_pred"],
#         "heroes_only_submission": fit_results["heroes_only_submission"],
#         "heroes_only_submission_path": fit_results["heroes_only_submission_path"],
#         "base_search_results": finalize_results["base_search_results"],
#         "mmr_comparison_results": finalize_results["mmr_comparison_results"],
#         "hero_model_results": finalize_results["hero_model_results"],
#         "cv_results": finalize_results["cv_results"],
#     }
#
