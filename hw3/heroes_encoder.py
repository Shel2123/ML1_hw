import numpy as np
import pandas as pd
import polars as pl


class HeroesEncoder:
    def __init__(self, heroes_df=None, split_teams=False, dtype=np.int8):
        self.n_features_ = None
        self.n_heroes_ = None
        self.hero_ids_ = None
        self.hero_to_col_ = None
        self.heroes_df = heroes_df
        self.split_teams = split_teams
        self.dtype = dtype
        self.is_fitted_ = False

    @staticmethod
    def _to_polars(df):
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(df, pl.LazyFrame):
            return df.collect()
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        if hasattr(df, "to_pandas"):
            return pl.from_pandas(df.to_pandas())
        return pl.from_pandas(pd.DataFrame(df))

    @classmethod
    def _to_lazy(cls, df) -> pl.LazyFrame:
        return df if isinstance(df, pl.LazyFrame) else cls._to_polars(df).lazy()

    @classmethod
    def _height(cls, df) -> int:
        if isinstance(df, pl.DataFrame):
            return df.height
        if isinstance(df, pl.LazyFrame):
            return df.select(pl.len().alias("n")).collect()["n"][0]
        return len(df)

    def fit(self, X, y=None):
        hero_source = self.heroes_df if self.heroes_df is not None else X
        hero_source_lf = self._to_lazy(hero_source)

        source_cols = set(hero_source_lf.collect_schema().names())
        if "id" in source_cols:
            hero_col = "id"
        elif "hero_id" in source_cols:
            hero_col = "hero_id"
        else:
            raise ValueError("В heroes_df нет колонки id или hero_id")

        hero_ids_df = (
            hero_source_lf
            .select(pl.col(hero_col).cast(pl.Int32, strict=False).alias("hero_id"))
            .drop_nulls()
            .unique()
            .sort("hero_id")
            .collect()
        )

        self.hero_ids_ = hero_ids_df["hero_id"].to_numpy()
        self.n_heroes_ = len(self.hero_ids_)
        self.n_features_ = self.n_heroes_ * 2 if self.split_teams else self.n_heroes_

        self.hero_to_col_ = pl.DataFrame({
            "hero_id": self.hero_ids_,
            "col": np.arange(self.n_heroes_, dtype=np.int32),
        })

        self.is_fitted_ = True
        return self

    def _build_players_df(self, X):
        if not self.is_fitted_:
            raise ValueError("Сначала fit")

        x_lf = self._to_lazy(X)

        players_lf = (
            x_lf
            .select([
                pl.col("match_id").cast(pl.Int64, strict=False).alias("match_id"),
                pl.col("hero_id").cast(pl.Int32, strict=False).alias("hero_id"),
                pl.col("player_slot").cast(pl.Int32, strict=False).alias("player_slot"),
            ])
            .filter(pl.col("hero_id").is_not_null() & (pl.col("hero_id") != 0))
            .join(self.hero_to_col_.lazy(), on="hero_id", how="left")
            .with_columns(
                pl.when(pl.col("player_slot").is_between(0, 4))
                .then(pl.lit(1, dtype=pl.Int8))
                .when(pl.col("player_slot").is_between(128, 132))
                .then(pl.lit(-1, dtype=pl.Int8))
                .otherwise(None)
                .alias("value")
            )
            .filter(pl.col("value").is_not_null())
        )

        if self.split_teams:
            players_lf = players_lf.with_columns([
                (
                    pl.col("col")
                    + pl.when(pl.col("value") == -1)
                    .then(pl.lit(self.n_heroes_, dtype=pl.Int32))
                    .otherwise(pl.lit(0, dtype=pl.Int32))
                ).alias("col"),
                pl.lit(1, dtype=pl.Int8).alias("value"),
            ])

        players_df = players_lf.select(["match_id", "col", "value"]).collect()

        if players_df["col"].null_count() > 0:
            raise ValueError("В players есть hero_id, которых нет в heroes_df.")

        return players_df

    def transform(self, X, matches_df=None, y=None):
        if not self.is_fitted_:
            raise ValueError("Сначала fit")

        players_df = self._build_players_df(X)

        if matches_df is None:
            matches_lf = (
                self._to_lazy(X)
                .select(pl.col("match_id").cast(pl.Int64, strict=False).alias("match_id"))
                .unique()
                .sort("match_id")
            )
            n_rows = self._height(matches_lf)
        else:
            matches_lf = self._to_lazy(matches_df).select(
                pl.col("match_id").cast(pl.Int64, strict=False).alias("match_id")
            )
            n_rows = self._height(matches_df)

        rows = (
            matches_lf
            .with_row_index("row")
            .join(players_df.lazy(), on="match_id", how="left")
            .drop_nulls(["col", "value"])
            .select(["row", "col", "value"])
            .collect()
        )

        row_idx = rows["row"].to_numpy()
        col_idx = rows["col"].to_numpy()
        data = rows["value"].to_numpy()

        matrix = np.zeros((n_rows, self.n_features_), dtype=self.dtype)

        if row_idx.size > 0:
            np.add.at(matrix, (row_idx, col_idx), data)

            if self.split_teams:
                np.clip(matrix, 0, 1, out=matrix)
            else:
                matrix = np.sign(matrix).astype(self.dtype, copy=False)

        return matrix

    def fit_transform(self, X, matches_df=None, y=None):
        self.fit(X, y)
        return self.transform(X, matches_df=matches_df)

    def transform_batches(self, X, matches_df=None, batch_size=100_000):
        n_rows = self._height(matches_df if matches_df is not None else X)
        for start in range(0, n_rows, batch_size):
            if matches_df is None:
                chunk = self._to_polars(X).slice(start, batch_size)
                yield self.transform(chunk)
            else:
                matches_chunk = self._to_polars(matches_df).slice(start, batch_size)
                yield self.transform(X, matches_df=matches_chunk)
