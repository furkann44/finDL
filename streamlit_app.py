from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_actions import STEP_LABELS, prepare_pipeline_commands, run_pipeline
from src.streamlit_auth import authenticated_user, init_auth_state, is_authenticated, load_auth_settings, logout, render_login_screen
from src.dashboard_data import (
    available_assets,
    available_models,
    build_asset_detail,
    build_recommendation_table,
    diagnostic_image_paths,
    ensure_window_columns,
    latest_prediction_snapshot,
    load_equity_curve,
    load_backtest_summary,
    load_holdout_summary,
    load_no_trade_summary,
    load_rolling_retrain_equity,
    load_rolling_retrain_summary,
    load_threshold_tuning_summary,
    load_walk_forward_baseline_summary,
    load_walk_forward_sequence_summary,
)


st.set_page_config(page_title="Financial Direction Dashboard", layout="wide")


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "total_return": "Total Return",
    "sharpe": "Sharpe",
    "recommended_total_return": "Recommended Total Return",
}


@st.cache_data(show_spinner=False)
def get_holdout_summary() -> pd.DataFrame:
    return load_holdout_summary(include_weighted=False)


@st.cache_data(show_spinner=False)
def get_backtest_summary() -> pd.DataFrame:
    return load_backtest_summary(include_weighted=False)


@st.cache_data(show_spinner=False)
def get_rolling_retrain_summary() -> pd.DataFrame:
    return load_rolling_retrain_summary()


@st.cache_data(show_spinner=False)
def get_threshold_tuning_summary() -> pd.DataFrame:
    return load_threshold_tuning_summary()


@st.cache_data(show_spinner=False)
def get_no_trade_summary(objective: str) -> pd.DataFrame:
    return load_no_trade_summary(optimize_for=objective)


@st.cache_data(show_spinner=False)
def get_recommendations() -> pd.DataFrame:
    return build_recommendation_table()


@st.cache_data(show_spinner=False)
def get_prediction_snapshot() -> pd.DataFrame:
    return latest_prediction_snapshot(models=None)


@st.cache_data(show_spinner=False)
def get_walk_forward_baseline() -> pd.DataFrame:
    return load_walk_forward_baseline_summary()


@st.cache_data(show_spinner=False)
def get_walk_forward_sequence() -> pd.DataFrame:
    return load_walk_forward_sequence_summary()


def filter_by_asset(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    if asset == "All Assets":
        return frame.copy()
    return frame[frame["symbol"] == asset].copy()


def render_metric_cards(recommendations: pd.DataFrame, selected_asset: str) -> None:
    filtered = filter_by_asset(recommendations, selected_asset)
    if filtered.empty:
        st.warning("Secili varlik icin onerilen konfigürasyon bulunamadi.")
        return

    if selected_asset == "All Assets":
        total_assets = filtered["symbol"].nunique()
        aligned_assets = int((filtered["recommendation_alignment"] == "aligned").sum())
        avg_holdout = filtered["holdout_roc_auc"].mean()
        avg_return = filtered["recommended_total_return"].mean()
        avg_rolling_return = filtered["best_rolling_total_return"].mean()
        cols = st.columns(5)
        cols[0].metric("Takip Edilen Varlik", total_assets)
        cols[1].metric("Uyusumlu Oneri", f"{aligned_assets}/{total_assets}")
        cols[2].metric("Ortalama Holdout ROC-AUC", f"{avg_holdout:.4f}")
        cols[3].metric("Ortalama Onerilen Return", f"{avg_return:.4f}")
        cols[4].metric("Ortalama Rolling Return", f"{avg_rolling_return:.4f}")
        return

    row = filtered.iloc[0]
    cols = st.columns(5)
    cols[0].metric("Onerilen Model", row["recommended_model"])
    cols[1].metric("Rolling Model", row["best_rolling_base_model"] if pd.notna(row["best_rolling_base_model"]) else "-" )
    cols[2].metric("Uyum", str(row["recommendation_alignment"]).upper())
    cols[3].metric("Onerilen Return", f"{row['recommended_total_return']:.4f}")
    cols[4].metric("Rolling Return", f"{row['best_rolling_total_return']:.4f}" if pd.notna(row["best_rolling_total_return"]) else "-")


def style_recommendation_table(frame: pd.DataFrame):
    def alignment_style(value: str) -> str:
        if value == "aligned":
            return "background-color: #dcfce7; color: #166534;"
        if value == "mixed":
            return "background-color: #fef3c7; color: #92400e;"
        return "background-color: #e5e7eb; color: #374151;"

    return frame.style.map(alignment_style, subset=["recommendation_alignment"])


def render_recommendations_tab(selected_asset: str) -> None:
    recommendations = get_recommendations()
    filtered = filter_by_asset(recommendations, selected_asset)

    st.subheader("Onerilen Konfigürasyon")
    render_metric_cards(recommendations, selected_asset)
    if selected_asset != "All Assets" and not filtered.empty:
        row = filtered.iloc[0]
        st.caption(
            "Onerilen test araligi: "
            f"{str(row['recommended_test_start'])[:10]} -> {str(row['recommended_test_end'])[:10]} | "
            f"Toplam test gunu: {int(row['recommended_test_rows'])} | "
            f"Aktif islem gunu: {int(row['recommended_active_trade_days'])}"
        )
    display_frame = filtered[
        [
            "symbol",
            "best_holdout_model",
            "holdout_roc_auc",
            "recommended_model",
            "recommended_total_return",
            "recommended_sharpe",
            "best_rolling_base_model",
            "best_rolling_total_return",
            "best_rolling_sharpe",
            "recommendation_alignment",
        ]
    ].copy()
    st.dataframe(style_recommendation_table(display_frame), use_container_width=True)


def render_holdout_tab(selected_asset: str, selected_models: list[str], selected_metric: str) -> None:
    holdout = get_holdout_summary()
    holdout = ensure_window_columns(holdout)
    holdout = holdout[holdout["split"] == "test"].copy()
    holdout = filter_by_asset(holdout, selected_asset)
    holdout = holdout[holdout["model"].isin(selected_models)].copy()

    st.subheader("Holdout Karsilastirmasi")
    if holdout.empty:
        st.info("Secilen filtrelerle holdout veri bulunamadi.")
        return

    chart_frame = holdout.pivot(index="symbol", columns="model", values=selected_metric)
    st.bar_chart(chart_frame)
    st.dataframe(holdout[["symbol", "model", "accuracy", "f1", "roc_auc", "predicted_positive_rate"]], use_container_width=True)


def render_no_trade_tab(selected_asset: str, selected_models: list[str], objective: str) -> None:
    no_trade = get_no_trade_summary(objective)
    no_trade = ensure_window_columns(no_trade)
    no_trade = filter_by_asset(no_trade, selected_asset)
    no_trade = no_trade[no_trade["model"].isin(selected_models)].copy()

    st.subheader("No-Trade Band Analizi")
    if no_trade.empty:
        st.info("Secilen filtrelerle no-trade veri bulunamadi.")
        return

    chart_frame = no_trade.pivot(index="symbol", columns="model", values="test_total_return")
    st.bar_chart(chart_frame)
    st.dataframe(
        no_trade[
            [
                "symbol",
                "model",
                "test_start",
                "test_end",
                "test_rows",
                "active_trade_days",
                "no_trade_days",
                "lower_threshold",
                "upper_threshold",
                "test_coverage",
                "test_active_f1",
                "test_total_return",
                "test_sharpe",
            ]
        ],
        use_container_width=True,
    )


def render_backtest_tab(selected_asset: str, selected_models: list[str]) -> None:
    backtest = get_backtest_summary()
    backtest = ensure_window_columns(backtest)
    backtest = filter_by_asset(backtest, selected_asset)
    backtest = backtest[backtest["model"].isin(selected_models)].copy()

    st.subheader("Backtest Ozetleri")
    if backtest.empty:
        st.info("Secilen filtrelerle backtest veri bulunamadi.")
        return

    chart_frame = backtest.pivot(index="symbol", columns="model", values="total_return")
    st.bar_chart(chart_frame)
    st.dataframe(
        backtest[["symbol", "model", "test_start", "test_end", "test_rows", "total_return", "benchmark_return", "sharpe", "max_drawdown", "win_rate"]],
        use_container_width=True,
    )


def render_rolling_retrain_tab(selected_asset: str) -> None:
    rolling = get_rolling_retrain_summary()
    rolling = filter_by_asset(rolling, selected_asset)

    st.subheader("Rolling Retraining")
    if rolling.empty:
        st.info("Rolling retraining verisi bulunamadi veya secili varlik icin henuz uretilmedi.")
        return

    chart_frame = rolling.pivot(index="symbol", columns="model", values="total_return")
    st.bar_chart(chart_frame)
    st.dataframe(
        rolling[
            [
                "symbol",
                "model",
                "base_model",
                "optimize_for",
                "cycles",
                "signal_rows",
                "test_start",
                "test_end",
                "coverage",
                "total_return",
                "benchmark_return",
                "sharpe",
                "max_drawdown",
                "active_accuracy",
                "active_f1",
            ]
        ],
        use_container_width=True,
    )


def render_threshold_tab(selected_asset: str, selected_models: list[str]) -> None:
    tuning = get_threshold_tuning_summary()
    tuning = filter_by_asset(tuning, selected_asset)
    tuning = tuning[tuning["model"].isin(selected_models)].copy()

    st.subheader("Threshold Tuning")
    if tuning.empty:
        st.info("Secilen filtrelerle threshold tuning veri bulunamadi.")
        return

    chart_frame = tuning.pivot(index="symbol", columns="model", values="test_f1_gain")
    st.bar_chart(chart_frame)
    st.dataframe(
        tuning[
            [
                "symbol",
                "model",
                "best_threshold",
                "test_f1_default",
                "test_f1_tuned",
                "test_f1_gain",
                "test_accuracy_default",
                "test_accuracy_tuned",
            ]
        ],
        use_container_width=True,
    )


def render_walk_forward_tab(selected_asset: str) -> None:
    baseline = filter_by_asset(get_walk_forward_baseline(), selected_asset)
    sequence = filter_by_asset(get_walk_forward_sequence(), selected_asset)

    st.subheader("Walk-Forward")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Walk-Forward**")
        if baseline.empty:
            st.info("Baseline walk-forward verisi bulunamadi.")
        else:
            st.dataframe(baseline, use_container_width=True)
    with col2:
        st.markdown("**Sequence Walk-Forward**")
        if sequence.empty:
            st.info("Sequence walk-forward verisi bulunamadi veya secili varlik icin uretilmedi.")
        else:
            st.dataframe(sequence, use_container_width=True)


def render_signals_tab(selected_asset: str, selected_models: list[str]) -> None:
    snapshot = get_prediction_snapshot()
    snapshot = filter_by_asset(snapshot, selected_asset)
    snapshot = snapshot[snapshot["model"].isin(selected_models)].copy()

    st.subheader("Son Skorlanan Ornekler")
    if snapshot.empty:
        st.info("Secilen filtrelerle signal snapshot bulunamadi.")
        return
    st.dataframe(snapshot, use_container_width=True)


def render_asset_detail_tab(selected_asset: str) -> None:
    st.subheader("Asset Detail")
    if selected_asset == "All Assets":
        st.info("Bu sekmede detay gormek icin tek bir varlik secin.")
        return

    detail = build_asset_detail(selected_asset)
    detail["holdout_summary"] = ensure_window_columns(detail["holdout_summary"])
    detail["no_trade_total_summary"] = ensure_window_columns(detail["no_trade_total_summary"])
    detail["backtest_summary"] = ensure_window_columns(detail["backtest_summary"])
    detail["rolling_summary"] = detail["rolling_summary"].copy()
    recommendation = detail["recommendation"]
    recommended_signal = detail["recommended_signal"]

    if recommendation is None:
        st.warning("Secili varlik icin detay veri bulunamadi.")
        return

    top_cols = st.columns(4)
    top_cols[0].metric("Onerilen Model", recommendation["recommended_model"])
    top_cols[1].metric("Rolling Model", recommendation["best_rolling_base_model"] if pd.notna(recommendation["best_rolling_base_model"]) else "-")
    top_cols[2].metric("Uyum", str(recommendation["recommendation_alignment"]).upper())
    top_cols[3].metric("No-Trade Return", f"{recommendation['recommended_total_return']:.4f}")

    st.caption(
        "Test araligi: "
        f"{str(recommendation['recommended_test_start'])[:10]} -> {str(recommendation['recommended_test_end'])[:10]} | "
        f"Toplam test gunu: {int(recommendation['recommended_test_rows'])} | "
        f"Aktif islem gunu: {int(recommendation['recommended_active_trade_days'])}"
    )

    if recommended_signal is not None:
        st.markdown("### Onerilen Son Sinyal")
        signal_cols = st.columns(5)
        signal_cols[0].metric("Yon", recommended_signal["direction"])
        signal_cols[1].metric("Olasilik", f"{recommended_signal['probability']:.4f}")
        signal_cols[2].metric("Lower Threshold", f"{recommended_signal['lower_threshold']:.2f}")
        signal_cols[3].metric("Upper Threshold", f"{recommended_signal['upper_threshold']:.2f}")
        signal_cols[4].metric("Tahmin Zamani", str(recommended_signal["datetime"])[:10])

    st.markdown("### Model Ozetleri")
    left, right = st.columns(2)
    with left:
        st.markdown("**Holdout Test**")
        st.dataframe(
            detail["holdout_summary"][["model", "accuracy", "f1", "roc_auc", "predicted_positive_rate", "test_start", "test_end", "test_rows"]],
            use_container_width=True,
        )
    with right:
        st.markdown("**No-Trade Total Return**")
        st.dataframe(
            detail["no_trade_total_summary"][[
                "model",
                "lower_threshold",
                "upper_threshold",
                "test_start",
                "test_end",
                "test_rows",
                "active_trade_days",
                "test_coverage",
                "test_total_return",
                "test_sharpe",
            ]],
            use_container_width=True,
        )

    st.markdown("### Equity Curve")
    equity_model = st.selectbox(
        "Equity curve modeli",
        options=detail["backtest_summary"]["model"].tolist(),
        key=f"equity_model_{selected_asset}",
    )
    equity_curve = load_equity_curve(selected_asset, equity_model)
    if {"datetime", "strategy_equity", "buy_hold_equity"}.issubset(equity_curve.columns):
        chart_frame = equity_curve[["datetime", "strategy_equity", "buy_hold_equity"]].copy().set_index("datetime")
        st.line_chart(chart_frame)
    st.dataframe(
        detail["backtest_summary"][["model", "test_start", "test_end", "test_rows", "total_return", "benchmark_return", "sharpe", "max_drawdown", "win_rate"]],
        use_container_width=True,
    )

    st.markdown("### Rolling Retraining")
    if detail["rolling_summary"].empty:
        st.info("Bu varlik icin rolling retraining sonucu henuz uretilmedi.")
    else:
        rolling_cols = st.columns(4)
        best_row = detail["rolling_summary"].sort_values("total_return", ascending=False).iloc[0]
        rolling_cols[0].metric("En Iyi Rolling Model", best_row["model"])
        rolling_cols[1].metric("Rolling Return", f"{best_row['total_return']:.4f}")
        rolling_cols[2].metric("Rolling Sharpe", f"{best_row['sharpe']:.4f}")
        rolling_cols[3].metric("Rolling Coverage", f"{best_row['coverage']:.4f}")

        rolling_model = st.selectbox(
            "Rolling equity modeli",
            options=detail["rolling_summary"]["model"].tolist(),
            key=f"rolling_model_{selected_asset}",
        )
        rolling_equity = load_rolling_retrain_equity(selected_asset, rolling_model)
        if {"datetime", "strategy_equity", "buy_hold_equity"}.issubset(rolling_equity.columns):
            rolling_chart = rolling_equity[["datetime", "strategy_equity", "buy_hold_equity"]].copy().set_index("datetime")
            st.line_chart(rolling_chart)
        st.dataframe(
            detail["rolling_summary"][
                [
                    "model",
                    "base_model",
                    "optimize_for",
                    "cycles",
                    "signal_rows",
                    "test_start",
                    "test_end",
                    "coverage",
                    "total_return",
                    "benchmark_return",
                    "sharpe",
                    "max_drawdown",
                    "active_accuracy",
                    "active_f1",
                ]
            ],
            use_container_width=True,
        )

    st.markdown("### ROC ve Confusion Matrix")
    diag_model = st.selectbox(
        "Diagnostik gorsel modeli",
        options=detail["holdout_summary"]["model"].tolist(),
        key=f"diag_model_{selected_asset}",
    )
    diagnostic_paths = diagnostic_image_paths(selected_asset, diag_model)
    diag_cols = st.columns(2)
    with diag_cols[0]:
        if diagnostic_paths["roc"]:
            st.image(str(diagnostic_paths["roc"]), caption=f"{selected_asset} {diag_model} ROC")
        else:
            st.info("ROC gorseli bulunamadi.")
    with diag_cols[1]:
        if diagnostic_paths["confusion"]:
            st.image(str(diagnostic_paths["confusion"]), caption=f"{selected_asset} {diag_model} Confusion Matrix")
        else:
            st.info("Confusion matrix gorseli bulunamadi.")


def render_management_tab(selected_asset: str) -> None:
    st.subheader("Web Uzerinden Yonetim")
    st.caption("Ham veri cekme, feature uretme, egitim ve ozet guncelleme adimlarini dashboard icinden calistirabilirsiniz.")

    if "management_results" not in st.session_state:
        st.session_state.management_results = []

    all_assets_scope = st.checkbox("Tum varliklarda calistir", value=(selected_asset == "All Assets"))
    if all_assets_scope:
        scope_symbols: list[str] = []
        st.info("Secili kapsam: tum tanimli varliklar")
    else:
        default_symbols = [] if selected_asset == "All Assets" else [selected_asset]
        scope_symbols = st.multiselect("Kapsamdaki varliklar", options=available_assets(), default=default_symbols)
        if not scope_symbols:
            st.warning("En az bir varlik secin veya tum varliklar secenegini kullanin.")

    st.markdown("### Hizli Aksiyonlar")
    quick_cols = st.columns(2)
    with quick_cols[0]:
        if st.button("Guncel Veri + Ozetleri Yenile", use_container_width=True):
            steps = ["fetch_raw", "build_processed", "summarize_results", "backtest", "threshold_tuning", "no_trade_total_return"]
            commands = prepare_pipeline_commands(scope_symbols, all_assets_scope, steps)
            with st.spinner("Guncelleme calistiriliyor..."):
                st.session_state.management_results = run_pipeline(commands)
                st.cache_data.clear()
    with quick_cols[1]:
        if st.button("Secili Varlik Icin Tam Pipeline", use_container_width=True):
            steps = [
                "fetch_raw",
                "build_processed",
                "train_baseline",
                "train_mlp",
                "train_lstm",
                "train_gru",
                "summarize_results",
                "backtest",
                "threshold_tuning",
                "no_trade_total_return",
            ]
            commands = prepare_pipeline_commands(scope_symbols, all_assets_scope, steps)
            with st.spinner("Tam pipeline calistiriliyor..."):
                st.session_state.management_results = run_pipeline(commands)
                st.cache_data.clear()

    st.markdown("### Ozel Pipeline")
    with st.form("custom_pipeline_form"):
        selected_steps = st.multiselect(
            "Adimlar",
            options=list(STEP_LABELS.keys()),
            default=["fetch_raw", "build_processed", "train_baseline", "summarize_results"],
            format_func=lambda key: STEP_LABELS[key],
        )
        col1, col2 = st.columns(2)
        with col1:
            lstm_epochs = st.number_input("LSTM epoch", min_value=1, max_value=100, value=12, step=1)
        with col2:
            gru_epochs = st.number_input("GRU epoch", min_value=1, max_value=100, value=12, step=1)
        submitted = st.form_submit_button("Pipeline Calistir")

    if submitted:
        if not all_assets_scope and not scope_symbols:
            st.error("Pipeline calistirmadan once en az bir varlik secin.")
        elif not selected_steps:
            st.error("En az bir adim secin.")
        else:
            commands = prepare_pipeline_commands(
                scope_symbols,
                all_assets_scope,
                selected_steps,
                lstm_epochs=int(lstm_epochs),
                gru_epochs=int(gru_epochs),
            )
            with st.spinner("Ozel pipeline calistiriliyor..."):
                st.session_state.management_results = run_pipeline(commands)
                st.cache_data.clear()

    if st.session_state.management_results:
        st.markdown("### Son Calistirma Sonuclari")
        summary_rows = [
            {
                "step": item["step"],
                "label": item["label"],
                "success": item["success"],
                "returncode": item["returncode"],
                "duration_seconds": round(item["duration_seconds"], 2),
            }
            for item in st.session_state.management_results
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        for item in st.session_state.management_results:
            status_icon = "success" if item["success"] else "error"
            with st.expander(f"{item['label']} | {status_icon} | {item['duration_seconds']:.2f}s"):
                st.code(item["command"])
                if item["stdout"]:
                    st.text_area("stdout", value=item["stdout"], height=180, key=f"stdout_{item['step']}")
                if item["stderr"]:
                    st.text_area("stderr", value=item["stderr"], height=160, key=f"stderr_{item['step']}")


def main() -> None:
    init_auth_state()
    auth_settings = load_auth_settings()
    if not is_authenticated():
        render_login_screen(auth_settings)
        return

    st.title("Financial Direction Dashboard")
    st.caption("Arastirma ciktilarini urun prototipine yaklastiran interaktif izleme ve karsilastirma arayuzu")

    assets = ["All Assets", *available_assets()]
    default_models = available_models(include_weighted=False)
    holdout_empty = get_holdout_summary().empty
    backtest_empty = get_backtest_summary().empty

    with st.sidebar:
        st.success(f"Giris yapan kullanici: {authenticated_user()}")
        if st.button("Cikis Yap", use_container_width=True):
            logout()
            st.rerun()
        st.divider()
        st.header("Filtreler")
        selected_asset = st.selectbox("Varlik", options=assets, index=0)
        selected_models = st.multiselect("Metotlar", options=default_models, default=default_models)
        selected_metric = st.selectbox(
            "Holdout Metrigi",
            options=["roc_auc", "f1", "accuracy"],
            format_func=lambda key: METRIC_LABELS[key],
        )
        no_trade_objective = st.selectbox(
            "No-Trade Objective",
            options=["active_f1", "total_return", "sharpe"],
        )

    if not selected_models:
        st.warning("En az bir model secin.")
        return

    if holdout_empty and backtest_empty:
        st.info(
            "Bu ortamda henuz uretilmis artifact bulunmuyor. Ilk kurulum icin `Management` sekmesinden veri ve model pipeline'ini calistirin."
        )

    tabs = st.tabs(
        [
            "Recommended Config",
            "Asset Detail",
            "Management",
            "Holdout",
            "No-Trade",
            "Backtest",
            "Rolling Retrain",
            "Threshold Tuning",
            "Walk-Forward",
            "Signals",
        ]
    )

    with tabs[0]:
        render_recommendations_tab(selected_asset)
    with tabs[1]:
        render_asset_detail_tab(selected_asset)
    with tabs[2]:
        render_management_tab(selected_asset)
    with tabs[3]:
        render_holdout_tab(selected_asset, selected_models, selected_metric)
    with tabs[4]:
        render_no_trade_tab(selected_asset, selected_models, no_trade_objective)
    with tabs[5]:
        render_backtest_tab(selected_asset, selected_models)
    with tabs[6]:
        render_rolling_retrain_tab(selected_asset)
    with tabs[7]:
        render_threshold_tab(selected_asset, selected_models)
    with tabs[8]:
        render_walk_forward_tab(selected_asset)
    with tabs[9]:
        render_signals_tab(selected_asset, selected_models)


if __name__ == "__main__":
    main()
