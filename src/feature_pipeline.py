import numpy as np
import pandas as pd


def process_user_features(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw chronological betting logs into a structured user-level

    behavioral matrix for risk model training and inference.

    Parameters:
        df_sorted (pd.DataFrame): Chronologically sorted betting transaction
        log.

    Returns:
        pd.DataFrame: Feature matrix aggregated per unique user_id.
    """
    print("Executing Behavioral Feature Engineering Pipeline...")

    # 1. Sequence Feature Generation (Lagged Variables)
    df_sorted["prev_bet_status"] = df_sorted.groupby("user_id")[
        "bet_status"
    ].shift(1)
    df_sorted["prev_fiat_bet_amount"] = df_sorted.groupby("user_id")[
        "fiat_bet_amount"
    ].shift(1)

    # 2. Bet size percentage change calculation
    df_sorted["bet_pct_change"] = (
        (df_sorted["fiat_bet_amount"] - df_sorted["prev_fiat_bet_amount"])
        / (df_sorted["prev_fiat_bet_amount"] + 1e-5)
    ) * 100
    df_sorted["bet_pct_change"] = (
        df_sorted["bet_pct_change"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Isolate bet spikes occurring immediately after losses
    df_sorted["loss_chase_val"] = np.where(
        df_sorted["prev_bet_status"] == "lose",
        df_sorted["bet_pct_change"],
        np.nan,
    )

    # Identify scalping behavior (odds <= 1.10x)
    df_sorted["is_scalp"] = (df_sorted["odds"] <= 1.10).astype(int)

    # 3. User Aggregations
    df_global = (
        df_sorted.groupby("user_id")
        .agg(
            total_bets=("bet_id", "count"),
            total_wagered=("fiat_bet_amount", "sum"),
            total_net_profit=("fiat_profit_amount", "sum"),
            total_scalp_bets=("is_scalp", "sum"),
            global_bet_sum=("fiat_bet_amount", "sum"),
            global_bet_sq_sum=("fiat_bet_amount", lambda x: np.sum(x**2)),
            global_loss_chase_sum=("loss_chase_val", "sum"),
            global_loss_count=("loss_chase_val", "count"),
        )
        .reset_index()
    )

    # 4. Behavioral Ratios 
    df_global["mean_bet"] = (
        df_global["global_bet_sum"] / df_global["total_bets"]
    )
    df_global["bet_variance"] = (
        df_global["global_bet_sq_sum"] / df_global["total_bets"]
    ) - (df_global["mean_bet"] ** 2)
    df_global["bet_std"] = np.sqrt(np.maximum(df_global["bet_variance"], 0))

    #  Psychological Metrics
    df_global["wager_volatility"] = df_global["bet_std"] / (
        df_global["mean_bet"] + 1e-5
    )
    df_global["loss_chase_ratio"] = df_global["global_loss_chase_sum"] / (
        df_global["global_loss_count"] + 1e-5
    )
    df_global["scalper_ratio"] = (
        df_global["total_scalp_bets"] / df_global["total_bets"]
    )

    # Clean intermediate calculation columns
    df_global.drop(
        columns=[
            "global_bet_sum",
            "global_bet_sq_sum",
            "global_loss_chase_sum",
            "global_loss_count",
            "mean_bet",
            "bet_variance",
            "bet_std",
            "total_scalp_bets",
        ],
        inplace=True,
    )

    # 5. Target Label Synthesis (Top 5% Net Losers = 1, Else = 0)
    df_global["net_loss"] = df_global["total_net_profit"].apply(
        lambda x: abs(x) if x < 0 else 0
    )
    loss_threshold = df_global["net_loss"].quantile(0.95)
    df_global["high_risk_target"] = (
        df_global["net_loss"] >= loss_threshold
    ).astype(int)

    return df_global


if __name__ == "__main__":
    print(
        "Pipeline script loaded."
    )
