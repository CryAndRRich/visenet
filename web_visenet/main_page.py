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

    # Khởi tạo session_state nếu chưa có
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "unread_count" not in st.session_state:
        st.session_state.unread_count = 0
    if "show_notif" not in st.session_state:
        st.session_state.show_notif = False

    # Hàm thêm thông báo mới
    def add_notification(message: str):
        st.session_state.notifications.insert(0, {
            "message": message,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "read": False
        })
        st.session_state.unread_count += 1

    # Kiểm tra xem có thông báo chưa đọc không
    has_unread = sum(notif["read"] == False for notif in st.session_state.notifications)

    # Nếu có thông báo chưa đọc -> in đậm
    expander_title = f"📜 Lịch sử thông báo ({has_unread})" if has_unread != 0 else "📜 Lịch sử thông báo"

    with st.expander(expander_title, expanded=False):
        # Tạo 2 cột cho 2 nút
        cols = st.columns([3, 2, 12], gap="small")

        with cols[0]:
            if st.button("Đánh dấu tất cả đã đọc", key="mark_read"):
                for n in st.session_state.notifications:
                    n["read"] = True
                st.session_state.unread_count = 0
                st.rerun()
                
        with cols[1]:
            if st.button("Xóa tất cả", key="clear_notif"):
                st.session_state.notifications.clear()
                st.session_state.unread_count = 0
                st.rerun()

        # Chỉ render khung nếu có thông báo
        if st.session_state.notifications:
            notif_html = "<div class='notif-box'>"
            for notif in st.session_state.notifications:
                color = "🔴" if not notif["read"] else "⚪"
                notif_html += f"<p>{color} <b>{notif['time']}</b><br>{notif['message']}</p>"
            notif_html += "</div>"

            st.markdown(notif_html, unsafe_allow_html=True)

    # if st.button("Thêm thông báo"):
    #     add_notification("Đây là thông báo mới!")

    cols = st.columns([1, 3])

    # ==========================================================
    # Load và tổng hợp dữ liệu từ file CSV
    # - Trả về giá đóng cửa của các cổ phiếu
    # - Tạo dataframe tổng hợp cho biểu đồ nến và các chỉ số kỹ thuật
    # ==========================================================
    @st.cache_resource(show_spinner=False)
    def load_and_aggregate(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Chuyển bytes thành file-like object
        df = pd.read_csv(io.BytesIO(file_bytes))

        df["Date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d")

        # Hàm tổng hợp
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
            st.error("❌ Lỗi khi đọc file")
            st.exception(e)
            st.stop()

        # Mặc định hiển thị 7 cổ phiếu đầu tiên
        STOCKS = sorted(df_agg["ticker"].unique().tolist())
        DEFAULT_STOCKS = STOCKS[:7]

    else:
        data_close, df_agg, STOCKS, DEFAULT_STOCKS = None, None, None, None
        st.warning("⚠️ Chưa có file thông tin mã cổ phiếu nào được tải lên")

    def stocks_to_str(stocks):
        return ",".join(stocks)

    if STOCKS is not None and DEFAULT_STOCKS is not None:
        # Khởi tạo tickers_input trong session_state nếu chưa có
        st.session_state.tickers_input = DEFAULT_STOCKS.copy()
            
        # ==========================================================
        # UI bên trái (bộ chọn cổ phiếu và phạm vi ngày)
        left = cols[0].container()

        with left:
            tickers = st.multiselect(
                "Mã cổ phiếu",
                options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
                default=st.session_state.tickers_input,
                placeholder="Chọn mã cổ phiếu để bắt đầu",
                help="Bạn có thể nhập mã cổ phiếu thủ công"
            )

        # ==========================================================
        # Chọn khoảng thời gian
        # ==========================================================
        with left:
            min_date = data_close.index.min().date()
            max_date = data_close.index.max().date()

            option = st.selectbox(
                "Chọn khoảng thời gian",
                ["1 tháng", "3 tháng", "6 tháng", 
                "1 năm", "2 năm", "5 năm", 
                "Toàn bộ thời gian",
                "Tự chọn"],
                index=0,
            )

            if option == "1 tháng":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif option == "3 tháng":
                start_date = max_date - timedelta(days=90)
                end_date = max_date
            elif option == "6 tháng":
                start_date = max_date - timedelta(days=180)
                end_date = max_date
            elif option == "1 năm":
                start_date = max_date - timedelta(days=365)
                end_date = max_date
            elif option == "2 năm":
                start_date = max_date - timedelta(days=2*365)
                end_date = max_date
            elif option == "5 năm":
                start_date = max_date - timedelta(days=5*365)
                end_date = max_date
            elif option == "Toàn bộ thời gian":
                start_date, end_date = min_date, max_date
            else:
                start_date = st.date_input(
                    "Ngày bắt đầu",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                )
                end_date = st.date_input(
                    "Ngày kết thúc",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                )
                if start_date > end_date:
                    st.error("⚠️ Ngày bắt đầu phải trước ngày kết thúc")
                    st.stop()

        # Mã cổ phiếu phải là chuỗi không rỗng và viết hoa
        tickers = [t.upper() for t in tickers if isinstance(t, str) and t.strip()]

        if tickers:
            st.query_params["stocks"] = stocks_to_str(tickers)
        else:
            st.query_params.pop("stocks", None)

        # Đảm bảo có ít nhất một mã cổ phiếu được chọn
        if not tickers:
            left.info("Chọn ít nhất một mã cổ phiểu", icon=":material/info:")
            st.stop()

        # ==========================================================
        # UI bên phải (biểu đồ và bảng)
        right = cols[1].container()

        # ==========================================================
        # Trường hợp 1: một cổ phiếu -> biểu đồ nến + chỉ số kỹ thuật
        # ==========================================================
        if len(tickers) == 1:
            ticker = tickers[0]

            df_t = df_agg[df_agg["ticker"] == ticker].set_index("Date").sort_index()
            df_t = df_t.loc[start_date:end_date].copy()

            if df_t.empty:
                st.error("Không có thông tin của mã trong khoảng thời gian đã chọn")
                st.stop()

            # Các cột chỉ số kỹ thuật
            indicator_cols = [c for c in df_t.columns if c not in ["ticker", "open", "high", "low", "close", "vol", "liq"]]

            # Người dùng chọn chỉ số để hiển thị
            default_inds = []
            selected_inds = right.multiselect(
                "Các chỉ số kỹ thuật",
                options=indicator_cols,
                default=default_inds,
                placeholder="Chọn các chỉ số kỹ thuật để hiển thị"
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

            # Thanh khoản (volume)
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

            st.markdown("**Dữ liệu gốc**")
            st.dataframe(df_t.reset_index(), width="stretch")

        # ==========================================================
        # Trường hợp 2: nhiều cổ phiếu -> biểu đồ giá chuẩn hóa + so sánh với trung bình các mã
        # ==========================================================
        else:
            try:
                data = data_close.loc[start_date:end_date, tickers]
            except Exception:
                tickers = [t for t in tickers if t in data_close.columns]
                if not tickers:
                    st.error("Không có thông tin của mã nào trong khoảng thời gian đã chọn")
                    st.stop()
                data = data_close.loc[start_date:end_date, tickers]

            if data.isna().all().any():
                empty_columns = data.columns[data.isna().all()].tolist()
                st.error(f"Lỗi khi tải dữ liệu: {', '.join(empty_columns)}.")
                st.stop()

            normalized = data.div(data.iloc[0])

            # Tính cổ phiếu tốt nhất và tệ nhất
            latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
            max_norm_value = max(latest_norm_values.items())
            min_norm_value = min(latest_norm_values.items()) 

            bottom = cols[0].container()
            with bottom:
                mcols = st.columns(2)
                mcols[0].metric(
                    "Tốt nhất",
                    max_norm_value[1],
                    delta=f"{round((max_norm_value[0] - 1) * 100, 2)}%",
                )
                mcols[1].metric(
                    "Tệ nhất",
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

            st.markdown("## So sánh với Trung bình các mã")
            if len(tickers) <= 1:
                st.warning("Cần chọn ít nhất 2 mã cổ phiếu để so sánh với Trung bình các mã", icon=":material/info:")
                st.stop()

            NUM_COLS = 2
            grid_cols = st.columns(NUM_COLS)
            for i, ticker in enumerate(tickers):
                peers = normalized.drop(columns=[ticker])
                peer_avg = peers.mean(axis=1)

                plot_data = pd.DataFrame({
                    "Date": normalized.index,
                    ticker: normalized[ticker],
                    "Trung bình các mã": peer_avg
                }).melt(id_vars=["Date"], var_name="Series", value_name="Price")

                chart1 = (
                    alt.Chart(plot_data)
                    .mark_line()
                    .encode(
                        alt.X("Date:T"),
                        alt.Y("Price:Q").scale(zero=False),
                        alt.Color("Series:N", scale=alt.Scale(domain=[ticker, "Trung bình các mã"], range=["red", "gray"])),
                        tooltip=["Date", "Series", "Price"],
                    )
                    .properties(title=f"{ticker} và Trung bình các mã", height=300)
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
                    .properties(title=f"Chênh lệch {ticker} với Trung bình các mã", height=300)
                )
                cell2 = grid_cols[(i * 2 + 1) % NUM_COLS].container()
                cell2.altair_chart(chart2)

            st.markdown("## Dữ liệu gốc")
            st.dataframe(data.reset_index(), width="stretch")
