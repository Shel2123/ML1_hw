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


FEATURE_GROUPS = {
    "region_ohe": FeatureGroup(ohe_cols=("region",)),
    "region_te": FeatureGroup(target_cols=("region",)),
    "game_mode_ohe": FeatureGroup(ohe_cols=("game_mode",)),
    "month_ohe": FeatureGroup(ohe_cols=("month",)),
    "is_weekend_ohe": FeatureGroup(ohe_cols=("is_weekend",)),
    "avg_mmr": FeatureGroup(numeric_cols=("avg_mmr", "mmr_missing")),
}


@dataclass(frozen=True)
class FeatureSelection:
    enabled: tuple[str, ...] = ()
    use_heroes: bool = False
    split_teams: bool = True
    mmr_transform: str = "sqrt"


def build_feature_config(selection: FeatureSelection) -> FeatureConfig:
    unknown = sorted(set(selection.enabled) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")

    target_cols: list[str] = []
    ohe_cols: list[str] = []
    numeric_cols: list[str] = []

    for name in unique_list(selection.enabled):
        group = FEATURE_GROUPS[name]
        target_cols.extend(group.target_cols)
        ohe_cols.extend(group.ohe_cols)
        numeric_cols.extend(group.numeric_cols)

    return FeatureConfig(
        target_cols=tuple(unique_list(target_cols)),
        ohe_cols=tuple(unique_list(ohe_cols)),
        numeric_cols=tuple(unique_list(numeric_cols)),
        mmr_transform=selection.mmr_transform,
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

        if not self.use_tabular and not self.use_heroes:
            raise ValueError("Нужно включить хотя бы одну фичу или heroes")

        self.n_splits = n_splits
        self.results_path = results_path
        self.submission_path = submission_path
        self.submission_id_col = submission_id_col
        self.target_col = target_col
        self.preprocess_players = preprocess_players
        self.model_params = model_params or {}

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

    def cross_validate(self, df_train, players_train=None, heroes_df=None):
        if self.use_heroes and players_train is None:
            raise ValueError("Для use_heroes=True нужен players_train")

        return base_cross_validate(
            df_train=df_train,
            cfg=self.cfg,
            players_train=players_train,
            use_tabular=self.use_tabular,
            use_heroes=self.use_heroes,
            heroes_df=heroes_df,
            split_teams=self.split_teams,
            n_splits=self.n_splits,
            target_col=self.target_col,
            model_params=self.model_params,
        )

    def fit_predict(self, df_train, df_test, players_train=None, players_test=None, heroes_df=None):
        if self.use_heroes and (players_train is None or players_test is None):
            raise ValueError("Для use_heroes=True нужны players_train и players_test")

        return fit_model_and_predict(
            df_train=df_train,
            df_test=df_test,
            cfg=self.cfg,
            players_train=players_train,
            players_test=players_test,
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

    def run(self, df_train, df_test, players_df=None, heroes_df=None):
        players_train, players_test = self._prepare_players(
            df_train=df_train,
            df_test=df_test,
            players_df=players_df,
        )

        mean_gini, fold_ginis = self.cross_validate(
            df_train=df_train,
            players_train=players_train,
            heroes_df=heroes_df,
        )

        model, test_pred = self.fit_predict(
            df_train=df_train,
            df_test=df_test,
            players_train=players_train,
            players_test=players_test,
            heroes_df=heroes_df,
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