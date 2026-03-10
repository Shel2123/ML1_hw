import json
import os
import warnings
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import polars as pl
import pandas as pd
import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cuml.linear_model import LogisticRegression as cuLogisticRegression
from heroes_encoder import HeroesEncoder


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*No categorical columns found.*")


@dataclass(frozen=True)
class FeatureConfig:
    target_cols: tuple[str, ...] = ()
    ohe_cols: tuple[str, ...] = ()
    numeric_cols: tuple[str, ...] = ()
    mmr_transform: str = "raw"

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


def to_polars_df(df):
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    if hasattr(df, "to_pandas"):
        return pl.from_pandas(df.to_pandas())
    return pl.from_pandas(pd.DataFrame(df))


def normalize_match_id(df: pl.DataFrame) -> pl.DataFrame:
    if "match_id" not in df.columns:
        return df
    return df.with_columns(
        pl.col("match_id").cast(pl.Int64, strict=False)
    )


def normalize_players_schema(df: pl.DataFrame) -> pl.DataFrame:
    columns = []
    if "match_id" in df.columns:
        columns.append(pl.col("match_id").cast(pl.Int64, strict=False))
    if "hero_id" in df.columns:
        columns.append(pl.col("hero_id").cast(pl.Int32, strict=False))
    if "player_slot" in df.columns:
        columns.append(pl.col("player_slot").cast(pl.Int32, strict=False))
    if not columns:
        return df
    return df.with_columns(columns)


def filter_players_by_match_ids(players_df, match_ids):
    players_pl = normalize_players_schema(to_polars_df(players_df))
    match_ids = np.asarray(pd.Index(match_ids).unique(), dtype=np.int64)
    match_ids_df = pl.DataFrame({"match_id": match_ids}).with_columns(
        pl.col("match_id").cast(pl.Int64, strict=False)
    )
    return players_pl.join(match_ids_df, on="match_id", how="inner")

def preprocess_players_df(players_df):
    players_pl = normalize_players_schema(to_polars_df(players_df))

    if "account_id" in players_pl.columns:
        players_pl = players_pl.filter(pl.col("account_id") != -1)
    required_cols = [col for col in ("match_id", "hero_id", "player_slot") if col in players_pl.columns]
    if required_cols:
        players_pl = players_pl.drop_nulls(subset=required_cols)

    bad_matches = (
        players_pl
        .group_by("match_id")
        .agg(
            pl.len().alias("rows_in_match"),
            pl.col("hero_id").n_unique().alias("unique_heroes"),
            (pl.col("hero_id") == 0).any().alias("has_hero0"),
        )
        .filter(
            (pl.col("rows_in_match") != 10) |
            (pl.col("unique_heroes") != 10) |
            pl.col("has_hero0")
        )
        .select("match_id")
        .unique()
    )

    if bad_matches.height > 0:
        players_pl = players_pl.join(bad_matches, on="match_id", how="anti")

    return players_pl.drop_nulls(["match_id", "hero_id", "player_slot"])

def split_players_by_matches(player_df, matches_train_df, matches_test_df):
    players_pl = normalize_players_schema(to_polars_df(player_df))
    train_matches_pl = normalize_match_id(to_polars_df(matches_train_df))
    test_matches_pl = normalize_match_id(to_polars_df(matches_test_df))

    train_ids = train_matches_pl.select("match_id").unique()
    test_ids = test_matches_pl.select("match_id").unique()

    overlap = train_ids.join(test_ids, on="match_id", how="inner").height
    if overlap > 0:
        raise ValueError("Один и тот же match_id попал и в train, и в test")

    players_train_df = players_pl.join(train_ids, on="match_id", how="inner")
    players_test_df = players_pl.join(test_ids, on="match_id", how="inner")

    return players_train_df, players_test_df


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

    return prep, X_train_block, y_train, X_other_block, y_other


def make_logreg(model_params=None):
    model_params = model_params or {}
    return cuLogisticRegression(
        penalty="l2",
        C=model_params.get("C", 1.0),
        max_iter=model_params.get("max_iter", 2000),
    )


def build_feature_matrices(
    train_matches,
    other_matches,
    cfg: FeatureConfig,
    *,
    train_players=None,
    other_players=None,
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    target_col="radiant_win",
):
    if not use_tabular and not use_heroes:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    X_train_blocks = []
    X_other_blocks = []

    y_train = (
        train_matches[target_col].astype(int).to_numpy()
        if target_col in train_matches.columns
        else None
    )
    y_other = (
        other_matches[target_col].astype(int).to_numpy()
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
            y_train = to_numpy(y_train_tab).astype(np.int32, copy=False)
        if y_other_tab is not None:
            y_other = to_numpy(y_other_tab).astype(np.int32, copy=False)

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

    return X_train, y_train, X_other, y_other

def as_float32(x):
    if x is None:
        return None
    if hasattr(x, "toarray"):
        arr = x.toarray()
    else:
        arr = to_numpy(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32, copy=False)


def combine_feature_blocks(*blocks):
    blocks = [as_float32(block) for block in blocks if block is not None]
    if not blocks:
        raise ValueError("Не передано ни одного блока фичей")
    if len(blocks) == 1:
        return blocks[0]
    return np.hstack(blocks).astype(np.float32, copy=False)


def unique_list(values):
    return list(dict.fromkeys(values))


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
    match mode:
        case "raw":
            df["avg_mmr"] = values
        case None:
            df["avg_mmr"] = values
        case "log1p":
            df["avg_mmr"] = np.log1p(values)
        case "sqrt":
            df["avg_mmr"] = np.sqrt(values)
        case _:
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


def cross_validate(
    df_train,
    cfg: FeatureConfig,
    *,
    players_train=None,
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    n_splits=5,
    target_col="radiant_win",
    model_params=None,
):
    if not use_tabular and not use_heroes:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    df_train = to_pandas_df(df_train)
    if use_heroes:
        players_train = to_polars_df(players_train)

    df_train["date"] = pd.to_datetime(df_train["date"])
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
                train_matches["match_id"].to_numpy(),
            )
            valid_players_fold = filter_players_by_match_ids(
                players_train,
                valid_matches["match_id"].to_numpy(),
            )

        X_train, y_train, X_valid, y_valid = build_feature_matrices(
            train_matches,
            valid_matches,
            cfg,
            train_players=train_players_fold,
            other_players=valid_players_fold,
            use_tabular=use_tabular,
            use_heroes=use_heroes,
            heroes_df=heroes_df,
            split_teams=split_teams,
            target_col=target_col,
        )

        model = make_logreg(model_params=model_params)
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
    use_tabular=True,
    use_heroes=False,
    heroes_df=None,
    split_teams=True,
    target_col="radiant_win",
    model_params=None,
):
    if not use_tabular and not use_heroes:
        raise ValueError("Нужно включить хотя бы один блок фичей")

    df_train = to_pandas_df(df_train)
    df_test = to_pandas_df(df_test)

    if use_heroes:
        players_train = to_polars_df(players_train)
        players_test = to_polars_df(players_test)

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_test["date"] = pd.to_datetime(df_test["date"])
    df_train = df_train.sort_values("date").reset_index(drop=True)

    if use_heroes and players_test.height == 0:
        warnings.warn("players_test пустой.")

    X_train, y_train, X_test, _ = build_feature_matrices(
        df_train,
        df_test,
        cfg,
        train_players=players_train,
        other_players=players_test,
        use_tabular=use_tabular,
        use_heroes=use_heroes,
        heroes_df=heroes_df,
        split_teams=split_teams,
        target_col=target_col,
    )

    if use_heroes and not np.any(X_test):
        warnings.warn("Фича героя для теста пустая.")

    model = make_logreg(model_params=model_params)
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


def prepare_players_for_run(players_train, players_test, df_train, df_test, preprocess_players=True):
    if not preprocess_players:
        return players_train, players_test

    players_all = pl.concat(
        [to_polars_df(players_train), to_polars_df(players_test)],
        how="vertical",
    )
    players_all = preprocess_players_df(players_all)
    return split_players_by_matches(players_all, df_train, df_test)


def log_result(results_store, stage, config_name, cfg: FeatureConfig, mean_gini, fold_ginis, model_name="cuML LogisticRegression"):
    results_store.append({
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "model": model_name,
        "features": config_name,
        "target_cols": list(cfg.target_cols),
        "ohe_cols": list(cfg.ohe_cols),
        "numeric_cols": list(cfg.numeric_cols),
        "mmr_transform": cfg.mmr_transform,
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


def stage_base_search(df_train, n_splits, target_col, results_store, base_configs=None):
    base_configs = build_base_configs() if base_configs is None else base_configs

    best_base_name = None
    best_base_cfg = None
    best_base_gini = -np.inf
    best_base_fold_ginis = None

    print("\nStage 1: search best base config without avg_mmr")

    for config_name, cfg in base_configs.items():
        print(f"\n{config_name}")

        mean_gini, fold_ginis = cross_validate(
            df_train=df_train,
            cfg=cfg,
            use_tabular=True,
            use_heroes=False,
            n_splits=n_splits,
            target_col=target_col,
        )

        log_result(
            results_store=results_store,
            stage="base_search",
            config_name=config_name,
            cfg=cfg,
            mean_gini=mean_gini,
            fold_ginis=fold_ginis,
            model_name="cuML LogisticRegression",
        )

        if mean_gini > best_base_gini:
            best_base_gini = mean_gini
            best_base_name = config_name
            best_base_cfg = cfg
            best_base_fold_ginis = fold_ginis

    print(f"\nBest base config: {best_base_name}")
    print(f"Best base mean Gini: {best_base_gini:.5f}")

    return {
        "best_base_name": best_base_name,
        "best_base_cfg": best_base_cfg,
        "best_base_gini": best_base_gini,
        "best_base_fold_ginis": best_base_fold_ginis,
    }


def stage_mmr_comparison(
    df_train,
    best_base_cfg,
    transformed_mmr,
    n_splits,
    target_col,
    results_store,
):
    comparison_configs = {
        f"base + {transformed_mmr} avg_mmr + mmr_missing": best_base_cfg.with_mmr(transformed_mmr),
    }

    comparison_scores = {}
    print("\nStage 2: compare base vs new MMR models")

    for config_name, cfg in comparison_configs.items():
        print(f"\n{config_name}")

        mean_gini, fold_ginis = cross_validate(
            df_train=df_train,
            cfg=cfg,
            use_tabular=True,
            use_heroes=False,
            n_splits=n_splits,
            target_col=target_col,
        )

        comparison_scores[config_name] = {
            "mean_gini": mean_gini,
            "fold_ginis": fold_ginis,
            "cfg": cfg,
        }

        log_result(
            results_store=results_store,
            stage="mmr_comparison",
            config_name=config_name,
            cfg=cfg,
            mean_gini=mean_gini,
            fold_ginis=fold_ginis,
            model_name="cuML LogisticRegression",
        )

    best_tabular_name = max(
        comparison_scores,
        key=lambda name: comparison_scores[name]["mean_gini"],
    )
    best_tabular_cfg = comparison_scores[best_tabular_name]["cfg"]

    print(f"\nBest tabular config after MMR comparison: {best_tabular_name}")
    print(f"Best tabular mean Gini: {comparison_scores[best_tabular_name]['mean_gini']:.5f}")

    return {
        "best_tabular_name": best_tabular_name,
        "best_tabular_cfg": best_tabular_cfg,
        "comparison_scores": comparison_scores,
    }


def stage_hero_models(
    df_train,
    players_train,
    heroes_df,
    split_teams,
    n_splits,
    target_col,
    best_tabular_cfg,
    results_store,
):
    hero_only_cfg = FeatureConfig()

    hero_model_specs = {
        "all features": {
            "cfg": best_tabular_cfg,
            "use_tabular": True,
            "use_heroes": True,
        },
        "heroes only": {
            "cfg": hero_only_cfg,
            "use_tabular": False,
            "use_heroes": True,
        },
    }

    hero_model_scores = {}
    print("\nStage 3: compare hero-based models")

    for config_name, spec in hero_model_specs.items():
        print(f"\n{config_name}")

        mean_gini, fold_ginis = cross_validate(
            df_train=df_train,
            cfg=spec["cfg"],
            players_train=players_train,
            use_tabular=spec["use_tabular"],
            use_heroes=spec["use_heroes"],
            heroes_df=heroes_df,
            split_teams=split_teams,
            n_splits=n_splits,
            target_col=target_col,
        )

        hero_model_scores[config_name] = {
            "mean_gini": mean_gini,
            "fold_ginis": fold_ginis,
            "cfg": spec["cfg"],
        }

        log_result(
            results_store=results_store,
            stage="hero_models",
            config_name=config_name,
            cfg=spec["cfg"],
            mean_gini=mean_gini,
            fold_ginis=fold_ginis,
            model_name="cuml logistic regression with heroes",
        )

    print("\nHero models comparison (CV)")
    print(f"All features Gini: {hero_model_scores['all features']['mean_gini']:.5f}")
    print(f"Heroes only Gini: {hero_model_scores['heroes only']['mean_gini']:.5f}")

    return hero_only_cfg, hero_model_scores


def stage_fit_hero_models_and_save(
    df_train,
    df_test,
    players_train,
    players_test,
    best_tabular_cfg,
    hero_only_cfg,
    heroes_df,
    split_teams,
    target_col,
    submission_path,
    submission_id_col,
):
    print("\nStage 4: fit full hero models and save test submissions")

    all_features_model, all_features_test_pred = fit_model_and_predict(
        df_train=df_train,
        df_test=df_test,
        players_train=players_train,
        players_test=players_test,
        cfg=best_tabular_cfg,
        use_tabular=True,
        use_heroes=True,
        heroes_df=heroes_df,
        split_teams=split_teams,
        target_col=target_col,
    )

    heroes_only_model, heroes_only_test_pred = fit_model_and_predict(
        df_train=df_train,
        df_test=df_test,
        players_train=players_train,
        players_test=players_test,
        cfg=hero_only_cfg,
        use_tabular=False,
        use_heroes=True,
        heroes_df=heroes_df,
        split_teams=split_teams,
        target_col=target_col,
    )

    all_features_submission_path = make_submission_path(submission_path, "all_features")
    heroes_only_submission_path = make_submission_path(submission_path, "heroes_only")

    all_features_submission = save_test_predictions(
        df_test=df_test,
        test_pred=all_features_test_pred,
        path=all_features_submission_path,
        id_col=submission_id_col,
    )

    heroes_only_submission = save_test_predictions(
        df_test=df_test,
        test_pred=heroes_only_test_pred,
        path=heroes_only_submission_path,
        id_col=submission_id_col,
    )

    return {
        "all_features_model": all_features_model,
        "all_features_test_pred": all_features_test_pred,
        "all_features_submission": all_features_submission,
        "all_features_submission_path": all_features_submission_path,
        "heroes_only_model": heroes_only_model,
        "heroes_only_test_pred": heroes_only_test_pred,
        "heroes_only_submission": heroes_only_submission,
        "heroes_only_submission_path": heroes_only_submission_path,
    }


def stage_finalize_results(results_store, results_path):
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_store, f, indent=2, ensure_ascii=False)

    cv_results = pd.DataFrame(results_store)
    base_search_results = cv_results[cv_results["stage"] == "base_search"].reset_index(drop=True)
    mmr_comparison_results = cv_results[cv_results["stage"] == "mmr_comparison"].reset_index(drop=True)
    hero_model_results = cv_results[cv_results["stage"] == "hero_models"].reset_index(drop=True)

    return {
        "cv_results": cv_results,
        "base_search_results": base_search_results,
        "mmr_comparison_results": mmr_comparison_results,
        "hero_model_results": hero_model_results,
    }


def run(
    df_train,
    df_test,
    players_df,
    heroes_df=None,
    split_teams=True,
    n_splits=5,
    results_path="results.json",
    submission_path="submission.csv",
    submission_id_col="match_id",
    target_col="radiant_win",
    transformed_mmr="sqrt",
    preprocess_players=True,
):
    results_store = []

    players_all = to_polars_df(players_df)
    if preprocess_players:
        players_all = preprocess_players_df(players_all)

    players_train, players_test = split_players_by_matches(players_all, df_train, df_test)

    base_results = stage_base_search(
        df_train=df_train,
        n_splits=n_splits,
        target_col=target_col,
        results_store=results_store,
    )

    mmr_results = stage_mmr_comparison(
        df_train=df_train,
        best_base_cfg=base_results["best_base_cfg"],
        transformed_mmr=transformed_mmr,
        n_splits=n_splits,
        target_col=target_col,
        results_store=results_store,
    )

    hero_only_cfg, hero_model_scores = stage_hero_models(
        df_train=df_train,
        players_train=players_train,
        heroes_df=heroes_df,
        split_teams=split_teams,
        n_splits=n_splits,
        target_col=target_col,
        best_tabular_cfg=mmr_results["best_tabular_cfg"],
        results_store=results_store,
    )

    fit_results = stage_fit_hero_models_and_save(
        df_train=df_train,
        df_test=df_test,
        players_train=players_train,
        players_test=players_test,
        best_tabular_cfg=mmr_results["best_tabular_cfg"],
        hero_only_cfg=hero_only_cfg,
        heroes_df=heroes_df,
        split_teams=split_teams,
        target_col=target_col,
        submission_path=submission_path,
        submission_id_col=submission_id_col,
    )

    finalize_results = stage_finalize_results(
        results_store=results_store,
        results_path=results_path,
    )

    return {
        "best_base_config": base_results["best_base_name"],
        "best_base_params": base_results["best_base_cfg"],
        "best_base_gini": base_results["best_base_gini"],
        "best_base_fold_ginis": base_results["best_base_fold_ginis"],
        "best_tabular_config_name": mmr_results["best_tabular_name"],
        "best_tabular_params": mmr_results["best_tabular_cfg"],
        "hero_model_scores": hero_model_scores,
        "all_features_model": fit_results["all_features_model"],
        "all_features_test_pred": fit_results["all_features_test_pred"],
        "all_features_submission": fit_results["all_features_submission"],
        "all_features_submission_path": fit_results["all_features_submission_path"],
        "heroes_only_model": fit_results["heroes_only_model"],
        "heroes_only_test_pred": fit_results["heroes_only_test_pred"],
        "heroes_only_submission": fit_results["heroes_only_submission"],
        "heroes_only_submission_path": fit_results["heroes_only_submission_path"],
        "base_search_results": finalize_results["base_search_results"],
        "mmr_comparison_results": finalize_results["mmr_comparison_results"],
        "hero_model_results": finalize_results["hero_model_results"],
        "cv_results": finalize_results["cv_results"],
    }

