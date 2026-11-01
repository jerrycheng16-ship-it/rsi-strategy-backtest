import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 網頁基本設定
st.set_page_config(
    page_title="雙重條件篩選策略回測系統", page_icon="📈", layout="wide"
)

st.title("📊 動態資產配置回測系統：高波動門檻 + 低 RSI 策略")
st.markdown("""
這個工具讓你可以自訂資產池，並在每個調倉日篩選出**年化波動度高於指定水準**且 **RSI 低於超賣閾值**的標的進行等權重配置；若無任何標的符合條件則當期空手。
""")


# 計算 RSI 的函數
def compute_rsi(series, period=14):
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


# 側邊欄：參數設定
st.sidebar.header("⚙️ 回測參數設定")

# 預設資產池
default_tickers = (
    "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, AMD, NFLX, QQQ, SPY"
)
tickers_input = st.sidebar.text_area(
    "輸入資產代號 (以逗號分隔)", default_tickers
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# 日期區間
start_date = st.sidebar.date_input(
    "開始日期", pd.to_datetime("2021-01-01")
)
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-01-01"))

# 策略參數
st.sidebar.subheader("策略細節設定")
rsi_threshold = st.sidebar.slider(
    "RSI 買入閾值 (低於此數值)", 10, 50, 35
)
vol_lookback = st.sidebar.slider(
    "波動度計算天數 (Lookback)", 10, 120, 60
)
min_vol_threshold = (
    st.sidebar.slider("最低年化波動度門檻 (%)", 10.0, 100.0, 30.0, 5.0) / 100.0
)

rebalance_freq = st.sidebar.selectbox(
    "調倉頻率", ["每月 (Monthly)", "每週 (Weekly)"]
)

run_button = st.sidebar.button("🚀 開始回測", type="primary")

if run_button:
  if len(tickers) == 0:
    st.error("錯誤：請至少輸入一個資產代號！")
  else:
    with st.spinner("正在下載歷史資料並進行回測運算中..."):
      # 1. 下載資料
      data = yf.download(tickers, start=start_date, end=end_date)[
          "Close"
      ]

      if isinstance(data, pd.Series):
        data = data.to_frame()

      data = data.dropna(axis=1, how="all")

      # 2. 計算每日報酬率、波動度與 RSI
      daily_returns = data.pct_change()
      volatility = daily_returns.rolling(window=vol_lookback).std() * np.sqrt(
          252
      )

      rsi_df = pd.DataFrame(index=data.index, columns=data.columns)
      for col in data.columns:
        rsi_df[col] = compute_rsi(data[col], period=14)

      # 3. 決定調倉日
      if "每月" in rebalance_freq:
        rebal_dates = data.resample("ME").last().index
      else:
        rebal_dates = data.resample("W").last().index

      rebal_dates = [d for d in rebal_dates if d >= data.index[vol_lookback]]

      portfolio_returns = []
      holdings_history = []
      all_period_details = []

      # 4. 回測主迴圈
      for i in range(len(rebal_dates) - 1):
        start_rd = rebal_dates[i]
        end_rd = rebal_dates[i + 1]

        if start_rd not in volatility.index:
          valid_idx = volatility.index.get_indexer(
              [start_rd], method="nearest"
          )[0]
          start_rd = volatility.index[valid_idx]

        current_vol = volatility.loc[start_rd]
        current_rsi = rsi_df.loc[start_rd]

        # 雙重條件篩選：年化波動度 >= 門檻 且 RSI < 閾值
        qualified = current_vol[
            (current_vol >= min_vol_threshold) & (current_rsi < rsi_threshold)
        ].index.tolist()

        selected = (
            qualified  # 符合條件的全數納入，若無則為空串列（代表空手）
        )

        holdings_history.append({
            "Date": start_rd.strftime("%Y-%m-%d"),
            "Selected Stocks": selected if selected else ["【無符合標的：空手】"],
        })

        # 記錄當期所有標的的檢視狀態
        for ticker in data.columns:
          v = current_vol[ticker]
          r = current_rsi[ticker]
          is_vol_ok = v >= min_vol_threshold if not pd.isna(v) else False
          is_rsi_ok = r < rsi_threshold if not pd.isna(r) else False
          is_selected = ticker in selected
          all_period_details.append({
              "調倉日期": start_rd.strftime("%Y-%m-%d"),
              "代號": ticker,
              "年化波動度": round(v, 4) if not pd.isna(v) else 0,
              "RSI 值": round(r, 2) if not pd.isna(r) else 0,
              f"符合波動門檻 (>= {min_vol_threshold*100:.0f}%)": (
                  "✅ 是" if is_vol_ok else "❌ 否"
              ),
              "符合 RSI 超賣": "✅ 是" if is_rsi_ok else "❌ 否",
              "是否入選配置": "⭐ 入選" if is_selected else "-",
          })

        # 計算報酬率（若有選中則等權重配置，若無選中則該期報酬為 0/空手）
        period_days_idx = daily_returns.loc[start_rd:end_rd].index[1:]
        if len(selected) > 0:
          period_data = daily_returns.loc[start_rd:end_rd, selected].iloc[
              1:
          ]
          if not period_data.empty:
            period_portfolio_return = period_data.mean(axis=1)
          else:
            period_portfolio_return = pd.Series(0.0, index=period_days_idx)
        else:
          # 當期空手，報酬率為 0
          period_portfolio_return = pd.Series(0.0, index=period_days_idx)

        if not period_portfolio_return.empty:
          portfolio_returns.append(period_portfolio_return)

      if portfolio_returns:
        strategy_rets = pd.concat(portfolio_returns)
        strategy_rets = strategy_rets[
            ~strategy_rets.index.duplicated(keep="first"
        )]

        benchmark_rets = daily_returns.mean(axis=1).loc[strategy_rets.index]

        # 累積報酬率
        strategy_cum = (1 + strategy_rets).cumprod()
        benchmark_cum = (1 + benchmark_rets).cumprod()

        # 策略績效指標
        total_return_strat = strategy_cum.iloc[-1] - 1
        annual_return_strat = (
            1 + total_return_strat
        ) ** (252 / len(strategy_rets)) - 1
        annual_vol_strat = strategy_rets.std() * np.sqrt(252)
        sharpe_ratio = (
            annual_return_strat / annual_vol_strat
            if annual_vol_strat != 0
            else 0
        )

        rolling_max_strat = strategy_cum.cummax()
        drawdown_strat = (strategy_cum - rolling_max_strat) / rolling_max_strat
        max_drawdown_strat = drawdown_strat.min()

        # 基準績效指標
        total_return_bench = benchmark_cum.iloc[-1] - 1
        annual_return_bench = (
            1 + total_return_bench
        ) ** (252 / len(benchmark_rets)) - 1
        annual_vol_bench = benchmark_rets.std() * np.sqrt(252)
        sharpe_ratio_bench = (
            annual_return_bench / annual_vol_bench
            if annual_vol_bench != 0
            else 0
        )

        rolling_max_bench = benchmark_cum.cummax()
        drawdown_bench = (benchmark_cum - rolling_max_bench) / rolling_max_bench
        max_drawdown_bench = drawdown_bench.min()

        # --- 呈現結果 ---
        st.success("回測完成！")

        # 1. 說明 Tag：策略選股模式解釋
        st.markdown("---")
        st.info(f"""
        📌 **【策略選股模式說明】**
        * **進場雙重條件**：在每個調倉日，資產必須同時滿足 **年化波動度 $\\ge$ {min_vol_threshold*100:.0f}%** 且 **RSI < {rsi_threshold}**。
        * **配置方式**：符合條件的所有標的進行**等權重配置**；若當期**沒有任何標的同時符合**上述兩個條件，則該期**選擇空手（持有現金，報酬率為 0）**，絕不勉強湊數。
        * **調倉頻率**：{rebalance_freq}。
        """)

        # 3. 核心績效數據指標呈現
        st.subheader("📊 核心績效數據總覽")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "策略年化報酬率",
            f"{annual_return_strat*100:.2f}%",
            f"總報酬: {total_return_strat*100:.2f}%",
        )
        col2.metric(
            "基準 (Benchmark) 年化報酬率",
            f"{annual_return_bench*100:.2f}%",
            f"總報酬: {total_return_bench*100:.2f}%",
        )
        col3.metric(
            "策略最大回撤 (MDD)",
            f"{max_drawdown_strat*100:.2f}%",
            f"夏普值: {sharpe_ratio:.2f}",
        )
        col4.metric(
            "基準最大回撤 (MDD)",
            f"{max_drawdown_bench*100:.2f}%",
            f"夏普值: {sharpe_ratio_bench:.2f}",
        )

        # 繪製績效曲線圖
        st.subheader("📈 策略 vs 基準 累積績效走勢")
        chart_data = pd.DataFrame(
            {
                "雙重條件策略 (含空手機制)": strategy_cum,
                "資產池等權重平均 (Benchmark)": benchmark_cum,
            }
        )
        st.line_chart(chart_data)

        # 4. 回撤圖表與表格
        st.subheader("📉 回撤 (Drawdown) 走勢分析")
        drawdown_chart_data = pd.DataFrame(
            {
                "策略回撤 (%)": drawdown_strat * 100,
                "基準回撤 (%)": drawdown_bench * 100,
            }
        )
        st.line_chart(drawdown_chart_data)

        dd_summary_df = pd.DataFrame({
            "指標": ["最大回撤 (Maximum Drawdown)", "平均回撤"],
            "策略": [
                f"{max_drawdown_strat*100:.2f}%",
                f"{drawdown_strat.mean()*100:.2f}%",
            ],
            "基準 (Benchmark)": [
                f"{max_drawdown_bench*100:.2f}%",
                f"{drawdown_bench.mean()*100:.2f}%",
            ],
        })
        st.dataframe(dd_summary_df, use_container_width=True)

        # 2. 每期詳細明細
        st.subheader("📋 每期調倉標的詳細明細（含雙重檢核狀態）")
        details_df = pd.DataFrame(all_period_details)
        st.dataframe(details_df, use_container_width=True)

      else:
        st.warning(
            "在選定的條件下沒有產生足夠的交易訊號，請嘗試調整波動度門檻或 RSI 閾值。"
        )
