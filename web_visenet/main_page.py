import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from typing import Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_main_page():
    st.set_page_config(
        page_title="VISENET",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )
    st.markdown("## VISENET")

    # Initialize session_state variables if they don't exist
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "unread_count" not in st.session_state:
        st.session_state.unread_count = 0
    if "show_notif" not in st.session_state:
        st.session_state.show_notif = False

    # Function to add a new notification
    def add_notification(message: str):
        st.session_state.notifications.insert(0, {
            "message": message,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "read": False
        })
        st.session_state.unread_count += 1

    # Check if there are any unread notifications
    has_unread = sum(notif["read"] == False for notif in st.session_state.notifications)

    # If there are unread notifications -> bold the title
    expander_title = f"📜 Notification history ({has_unread})" if has_unread != 0 else "📜 Notification history"

    with st.expander(expander_title, expanded=False):
        cols = st.columns([3, 2, 12], gap="small")

        with cols[0]:
            if st.button("Mark as read", key="mark_read"):
                for n in st.session_state.notifications:
                    n["read"] = True
                st.session_state.unread_count = 0
                st.rerun()
                
        with cols[1]:
            if st.button("Clear all", key="clear_notif"):
                st.session_state.notifications.clear()
                st.session_state.unread_count = 0
                st.rerun()

        # Only render the notification box if there are notifications
        if st.session_state.notifications:
            notif_html = "<div class='notif-box'>"
            for notif in st.session_state.notifications:
                color = "🔴" if not notif["read"] else "⚪"
                notif_html += f"<p>{color} <b>{notif['time']}</b><br>{notif['message']}</p>"
            notif_html += "</div>"

            st.markdown(notif_html, unsafe_allow_html=True)

    cols = st.columns([1, 3])

    # ==========================================================
    # Load and aggregate data from CSV file
    # - Return closing prices of stocks
    # - Create aggregated dataframe for candlestick chart and technical indicators
    # ==========================================================
    @st.cache_resource(show_spinner=False)
    def load_and_aggregate(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and aggregate stock data from CSV file bytes
        
        Parameters:
            file_bytes: bytes of the uploaded CSV file
        
        Returns:
            data_close: DataFrame of closing prices with Date as index and tickers as columns
            df_agg: Aggregated DataFrame with OHLC and technical indicators
        """
        # Change bytes to file-like object
        df = pd.read_csv(io.BytesIO(file_bytes))

        df["Date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d")

        # Functions to aggregate data
        agg_funcs = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "vol": "sum",
            "liq": "mean",
            "rsi": "mean",
            "macd": "mean",
            "cci": "mean",
            "adx": "mean",
            "turbulence": "mean",
        }

        df_agg = (
            df.groupby(["Date", "ticker"], as_index=False)
            .agg(agg_funcs)
            .sort_values(["ticker", "Date"])
        )

        data_close = df_agg.pivot(index="Date", columns="ticker", values="close").sort_index()

        return data_close, df_agg

    if "uploaded_file_bytes" in st.session_state:
        try:
            data_close, df_agg = load_and_aggregate(st.session_state["uploaded_file_bytes"])
        except Exception as e:
            st.error("❌ Error reading file")
            st.exception(e)
            st.stop()

        # Default to show the first 7 stocks
        STOCKS = sorted(df_agg["ticker"].unique().tolist())
        DEFAULT_STOCKS = STOCKS[:7]

    else:
        data_close, df_agg, STOCKS, DEFAULT_STOCKS = None, None, None, None
        st.warning("⚠️ No stock data file has been uploaded yet")

    def stocks_to_str(stocks):
        return ",".join(stocks)

    if STOCKS is not None and DEFAULT_STOCKS is not None:
        # Initialize tickers_input in session_state if not exists
        st.session_state.tickers_input = DEFAULT_STOCKS.copy()
            
        # ==========================================================
        # Left UI (stock selector and date range)
        left = cols[0].container()

        with left:
            tickers = st.multiselect(
                "Stock Tickers",
                options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
                default=st.session_state.tickers_input,
                placeholder="Choose stock tickers to start",
                help="You can manually input stock tickers"
            )

        # ==========================================================
        # Choose date range
        # ==========================================================
        with left:
            min_date = data_close.index.min().date()
            max_date = data_close.index.max().date()

            option = st.selectbox(
                "Choose date range",
                ["1 month", "3 months", "6 months", 
                "1 year", "2 years", "5 years", 
                "All time",
                "Custom"],
                index=0,
            )

            if option == "1 month":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif option == "3 months":
                start_date = max_date - timedelta(days=90)
                end_date = max_date
            elif option == "6 months":
                start_date = max_date - timedelta(days=180)
                end_date = max_date
            elif option == "1 year":
                start_date = max_date - timedelta(days=365)
                end_date = max_date
            elif option == "2 years":
                start_date = max_date - timedelta(days=2*365)
                end_date = max_date
            elif option == "5 years":
                start_date = max_date - timedelta(days=5*365)
                end_date = max_date
            elif option == "All time":
                start_date, end_date = min_date, max_date
            else:
                start_date = st.date_input(
                    "Start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                )
                end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                )
                if start_date > end_date:
                    st.error("⚠️ Start date must be before end date")
                    st.stop()

        # Stock tickers must be non-empty strings and uppercase
        tickers = [t.upper() for t in tickers if isinstance(t, str) and t.strip()]

        if tickers:
            st.query_params["stocks"] = stocks_to_str(tickers)
        else:
            st.query_params.pop("stocks", None)

        # Make sure at least one stock ticker is selected
        if not tickers:
            left.info("Choose at least one stock ticker", icon=":material/info:")
            st.stop()

        # ==========================================================
        # Right UI (charts and tables)
        right = cols[1].container()

        # ==========================================================
        # Case 1: single stock -> candlestick chart + technical indicators
        # ==========================================================
        if len(tickers) == 1:
            ticker = tickers[0]

            df_t = df_agg[df_agg["ticker"] == ticker].set_index("Date").sort_index()
            df_t = df_t.loc[start_date:end_date].copy()

            if df_t.empty:
                st.error("No information of the ticker in the selected date range")
                st.stop()

            # Technical indicator columns
            indicator_cols = [c for c in df_t.columns if c not in ["ticker", "open", "high", "low", "close", "vol", "liq"]]

            # User can select which indicators to display
            default_inds = []
            selected_inds = right.multiselect(
                "Technical Indicators",
                options=indicator_cols,
                default=default_inds,
                placeholder="Choose technical indicators to display"
            )

            rows = 1 + len(selected_inds)
            specs = [[{"secondary_y": True}]] + [[{}] for _ in selected_inds]

            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5] + [0.5 / max(1, len(selected_inds)) for _ in selected_inds],
                specs=specs,
            )

            fig.add_trace(
                go.Candlestick(
                    x=df_t.index,
                    open=df_t["open"],
                    high=df_t["high"],
                    low=df_t["low"],
                    close=df_t["close"],
                    name=f"{ticker} OHLC",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

            # Volume as bar chart on secondary y-axis
            if "vol" in df_t.columns:
                fig.add_trace(
                    go.Bar(
                        x=df_t.index,
                        y=df_t["vol"],
                        name="Volume",
                        opacity=0.5,
                        marker=dict(color="#606c76"),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                    secondary_y=True,
                )
                fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)

            for sma_col in [c for c in df_t.columns if c.lower().startswith("sma") or c.lower().startswith("ema")]:
                fig.add_trace(
                    go.Scatter(
                        x=df_t.index,
                        y=df_t[sma_col],
                        mode="lines",
                        line=dict(width=1.5),
                        name=sma_col,
                    ),
                    row=1,
                    col=1,
                    secondary_y=False,
                )

            for i, ind in enumerate(selected_inds, start=2):
                if ind not in df_t.columns:
                    continue
                fig.add_trace(
                    go.Scatter(x=df_t.index, y=df_t[ind], mode="lines", name=ind),
                    row=i,
                    col=1,
                )
                if ind.lower() in ("macd", "cci"):
                    fig.add_trace(
                        go.Scatter(x=df_t.index, y=[0] * len(df_t), mode="lines", line=dict(color="#888", dash="dash"), showlegend=False),
                        row=i,
                        col=1,
                    )
                fig.update_yaxes(title_text=ind.upper(), row=i, col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=300 + 200 * max(0, len(selected_inds)),
                margin=dict(l=10, r=10, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            right.plotly_chart(fig, use_container_width=True)

            st.markdown("## Raw Data")
            st.dataframe(df_t.reset_index(), width="stretch")

        # ==========================================================
        # Case 2: multiple stocks -> normalized price chart + compare with peer average
        # ==========================================================
        else:
            try:
                data = data_close.loc[start_date:end_date, tickers]
            except Exception:
                tickers = [t for t in tickers if t in data_close.columns]
                if not tickers:
                    st.error("No information of the tickers in the selected date range")
                    st.stop()
                data = data_close.loc[start_date:end_date, tickers]

            if data.isna().all().any():
                empty_columns = data.columns[data.isna().all()].tolist()
                st.error(f"Error loading data for: {', '.join(empty_columns)}")
                st.stop()

            normalized = data.div(data.iloc[0])

            # Calculate best and worst performing stocks
            latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
            max_norm_value = max(latest_norm_values.items())
            min_norm_value = min(latest_norm_values.items()) 

            bottom = cols[0].container()
            with bottom:
                mcols = st.columns(2)
                mcols[0].metric(
                    "Best",
                    max_norm_value[1],
                    delta=f"{round((max_norm_value[0] - 1) * 100, 2)}%",
                )
                mcols[1].metric(
                    "Worst",
                    min_norm_value[1],
                    delta=f"{round((min_norm_value[0] - 1) * 100, 2)}%",
                )

            chart_data = normalized.reset_index().melt(id_vars=["Date"], var_name="Stock", value_name="Normalized price")
            chart = (
                alt.Chart(chart_data)
                .mark_line()
                .encode(
                    alt.X("Date:T"),
                    alt.Y("Normalized price:Q").scale(zero=False),
                    alt.Color("Stock:N"),
                    tooltip=["Date", "Stock", alt.Tooltip("Normalized price", format=".4f")],
                )
                .properties(height=420)
            )
            right.altair_chart(chart)

            st.markdown("## Compare with Peer Average")
            if len(tickers) <= 1:
                st.warning("Need to select at least 2 stock tickers to compare with Peer Average", icon=":material/info:")
                st.stop()

            NUM_COLS = 2
            grid_cols = st.columns(NUM_COLS)
            for i, ticker in enumerate(tickers):
                peers = normalized.drop(columns=[ticker])
                peer_avg = peers.mean(axis=1)

                plot_data = pd.DataFrame({
                    "Date": normalized.index,
                    ticker: normalized[ticker],
                    "Peer Average": peer_avg
                }).melt(id_vars=["Date"], var_name="Series", value_name="Price")

                chart1 = (
                    alt.Chart(plot_data)
                    .mark_line()
                    .encode(
                        alt.X("Date:T"),
                        alt.Y("Price:Q").scale(zero=False),
                        alt.Color("Series:N", scale=alt.Scale(domain=[ticker, "Peer Average"], range=["red", "gray"])),
                        tooltip=["Date", "Series", "Price"],
                    )
                    .properties(title=f"{ticker} và Peer Average", height=300)
                )

                cell = grid_cols[(i * 2) % NUM_COLS].container()
                cell.altair_chart(chart1)

                delta_df = pd.DataFrame({"Date": normalized.index, "Delta": normalized[ticker] - peer_avg})
                chart2 = (
                    alt.Chart(delta_df)
                    .mark_area()
                    .encode(
                        alt.X("Date:T"),
                        alt.Y("Delta:Q").scale(zero=False),
                        tooltip=["Date", "Delta"],
                    )
                    .properties(title=f"Difference of {ticker} vs Peer Average", height=300)
                )
                cell2 = grid_cols[(i * 2 + 1) % NUM_COLS].container()
                cell2.altair_chart(chart2)

            st.markdown("## Raw Data")
            st.dataframe(data.reset_index(), width="stretch")
