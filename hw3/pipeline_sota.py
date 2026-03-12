import json
from dataclasses import asdict, dataclass

from hw3.pipeline3 import (
    FeatureConfig,
    cross_validate as base_cross_validate,
    fit_model_and_predict,
    prepare_players_for_run,
    save_test_predictions,
    split_players_by_matches,
    unique_list,
)


@dataclass(frozen=True)
class FeatureGroup:
    target_cols: tuple[str, ...] = ()
    ohe_cols: tuple[str, ...] = ()
    numeric_cols: tuple[str, ...] = ()
    text_cols: tuple[str, ...] = ()


FEATURE_GROUPS = {
    "region_ohe": FeatureGroup(ohe_cols=("region",)),
    "region_te": FeatureGroup(target_cols=("region",)),
    "game_mode_ohe": FeatureGroup(ohe_cols=("game_mode",)),
    "month_ohe": FeatureGroup(ohe_cols=("month",)),
    "is_weekend_ohe": FeatureGroup(ohe_cols=("is_weekend",)),
    "avg_mmr": FeatureGroup(numeric_cols=("avg_mmr", "mmr_missing")),
    "chat_tfidf": FeatureGroup(text_cols=("radiant_chat", "dire_chat")),
    "aggregates_gold_rad": FeatureGroup(numeric_cols=(
        'std_gold_adv',
        'abs_max_gold_adv',
        'peak_radiant_gold_adv',
         'radiant_mean_gold_adv',
         'radiant_last_gold_adv',
    )),
    "aggregates_gold_dire": FeatureGroup(numeric_cols=(
        'peak_dire_gold_adv',
        'dire_mean_gold_adv',
        'dire_last_gold_adv',

    )),
    "aggegates_exp_rad": FeatureGroup(numeric_cols=(
        'std_exp_adv',
        'peak_radiant_exp_adv',
        'abs_max_exp_adv',
        'radiant_mean_exp_adv',
        'radiant_last_exp_adv',

    )),
    "aggregates_exp_dire": FeatureGroup(numeric_cols=(
        'peak_dire_exp_adv',
        'dire_mean_exp_adv',
        'dire_last_exp_adv'

    )),
    "aggregates_trend_slope": FeatureGroup(numeric_cols=(
        'radiant_gold_adv_slope',
        'radiant_exp_adv_slope',

    )),
    "aggregates_trend_r2": FeatureGroup(numeric_cols=(
        'radiant_gold_adv_r2',
        'radiant_exp_adv_r2',

    )),
    "aggregates_trend_intercept": FeatureGroup(numeric_cols=(
        'radiant_gold_adv_intercept',
        'radiant_exp_adv_intercept',

    )),
    'trend_was_imputed': FeatureGroup(numeric_cols=('trend_was_imputed',)),

    "features_bin_gold_time": FeatureGroup(numeric_cols=(
        'gold_bin_m5',
         'gold_bin_m10',
         'gold_bin_m15',
    )),
    "features_bin_exp_time": FeatureGroup(numeric_cols=(
         'exp_bin_m5',
         'exp_bin_m10',
         'exp_bin_m15',
    )),
    "features_bin_gold_desc": FeatureGroup(numeric_cols=(
         'gold_adv_mean',
         'gold_adv_std',
         'gold_adv_min',
         'gold_adv_max',
    )),
    "features_bin_exp_desc": FeatureGroup(numeric_cols=(
         'exp_adv_mean',
         'exp_adv_std',
         'exp_adv_min',
         'exp_adv_max',
    )),
    "features_bin_gold_trend": FeatureGroup(numeric_cols=(
         'share_gold_positive',
         'share_exp_positive',
         'gold_bin_last',
         'exp_bin_last'
    ))
}


@dataclass(frozen=True)
class FeatureSelection:
    enabled: tuple[str, ...] = ()
    use_heroes: bool = False
    split_teams: bool = True
    mmr_transform: str = "sqrt"
    text_max_features: int = 3000
    text_min_df: int | float = 10
    text_max_df: int | float = 0.8
    text_ngram_range: tuple[int, int] = (1, 2)


def build_feature_config(selection: FeatureSelection) -> FeatureConfig:
    unknown = sorted(set(selection.enabled) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")

    target_cols: list[str] = []
    ohe_cols: list[str] = []
    numeric_cols: list[str] = []
    text_cols: list[str] = []

    for name in unique_list(selection.enabled):
        group = FEATURE_GROUPS[name]
        target_cols.extend(group.target_cols)
        ohe_cols.extend(group.ohe_cols)
        numeric_cols.extend(group.numeric_cols)
        text_cols.extend(group.text_cols)

    return FeatureConfig(
        target_cols=tuple(unique_list(target_cols)),
        ohe_cols=tuple(unique_list(ohe_cols)),
        numeric_cols=tuple(unique_list(numeric_cols)),
        text_cols=tuple(unique_list(text_cols)),
        mmr_transform=selection.mmr_transform,
        text_max_features=selection.text_max_features,
        text_min_df=selection.text_min_df,
        text_max_df=selection.text_max_df,
        text_ngram_range=selection.text_ngram_range,
    )


class MyPipeline:
    def __init__(
        self,
        feature_selection: FeatureSelection,
        *,
        n_splits: int = 5,
        results_path: str = "results.json",
        submission_path: str = "submission.csv",
        submission_id_col: str = "match_id",
        target_col: str = "radiant_win",
        preprocess_players: bool = True,
        model_params: dict | None = None,
    ):
        self.feature_selection = feature_selection
        self.cfg = build_feature_config(feature_selection)

        self.use_tabular = any((
            self.cfg.target_cols,
            self.cfg.ohe_cols,
            self.cfg.numeric_cols,
        ))
        self.use_heroes = feature_selection.use_heroes
        self.split_teams = feature_selection.split_teams

        self.n_splits = n_splits
        self.results_path = results_path
        self.submission_path = submission_path
        self.submission_id_col = submission_id_col
        self.target_col = target_col
        self.preprocess_players = preprocess_players
        self.model_params = model_params or {}
        self.use_text = bool(self.cfg.text_cols)

        if not self.use_tabular and not self.use_heroes and not self.use_text:
            raise ValueError("Нужно включить хотя бы один блок фичей")


    def _prepare_players(self, df_train, df_test, players_df):
        if not self.use_heroes:
            return None, None

        if players_df is None:
            raise ValueError("Для use_heroes=True нужен players_df")

        players_train, players_test = split_players_by_matches(
            players_df,
            df_train,
            df_test,
        )

        players_train, players_test = prepare_players_for_run(
            players_train,
            players_test,
            df_train,
            df_test,
            preprocess_players=self.preprocess_players,
        )

        return players_train, players_test

    def cross_validate(self, df_train, players_train=None, heroes_df=None, chat_df=None):
        if self.use_heroes and players_train is None:
            raise ValueError("Для use_heroes=True нужен players_train")

        return base_cross_validate(
            df_train=df_train,
            cfg=self.cfg,
            players_train=players_train,
            chat_df=chat_df,
            use_tabular=self.use_tabular,
            use_heroes=self.use_heroes,
            heroes_df=heroes_df,
            split_teams=self.split_teams,
            n_splits=self.n_splits,
            target_col=self.target_col,
            model_params=self.model_params,
        )

    def fit_predict(self, df_train, df_test, players_train=None, players_test=None, heroes_df=None, chat_df=None):
        if self.use_heroes and (players_train is None or players_test is None):
            raise ValueError("Для use_heroes=True нужны players_train и players_test")

        return fit_model_and_predict(
            df_train=df_train,
            df_test=df_test,
            cfg=self.cfg,
            players_train=players_train,
            players_test=players_test,
            chat_df=chat_df,
            use_tabular=self.use_tabular,
            use_heroes=self.use_heroes,
            heroes_df=heroes_df,
            split_teams=self.split_teams,
            target_col=self.target_col,
            model_params=self.model_params,
        )

    def _save_run_meta(self, mean_gini, fold_ginis):
        payload = {
            "selected_feature_groups": list(self.feature_selection.enabled),
            "use_heroes": self.use_heroes,
            "split_teams": self.split_teams,
            "feature_config": asdict(self.cfg),
            "model_params": self.model_params,
            "cv_mean_gini": round(float(mean_gini), 6),
            "cv_fold_ginis": [round(float(x), 6) for x in fold_ginis],
        }

        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return payload

    def run(self, df_train, df_test, players_df=None, heroes_df=None, chat_df=None):
        players_train, players_test = self._prepare_players(
            df_train=df_train,
            df_test=df_test,
            players_df=players_df,
        )

        mean_gini, fold_ginis = self.cross_validate(
            df_train=df_train,
            players_train=players_train,
            heroes_df=heroes_df,
            chat_df=chat_df,
        )

        model, test_pred = self.fit_predict(
            df_train=df_train,
            df_test=df_test,
            players_train=players_train,
            players_test=players_test,
            heroes_df=heroes_df,
            chat_df=chat_df,
        )

        submission = save_test_predictions(
            df_test=df_test,
            test_pred=test_pred,
            path=self.submission_path,
            id_col=self.submission_id_col,
        )

        run_meta = self._save_run_meta(mean_gini, fold_ginis)

        return {
            "model": model,
            "test_pred": test_pred,
            "submission": submission,
            "submission_path": self.submission_path,
            "cv_mean_gini": mean_gini,
            "cv_fold_ginis": fold_ginis,
            "run_meta": run_meta,
        }
