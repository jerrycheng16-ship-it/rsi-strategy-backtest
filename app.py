import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 網頁基本設定
st.set_page_config(
    page_title="高波動低 RSI 策略回測系統", page_icon="📈", layout="wide"
)

st.title("📊 動態資產配置回測系統：高波動 + 低 RSI 策略")
st.markdown("""
這個工具讓你可以自訂資產池，並在每個調倉日篩選出**波動度較高**且 **RSI 低於指定水準**（超賣現象）的標的進行等權重配置，並回測其歷史績效。
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

# 預設資產池（美股科技與熱門股，台股可加上 .TW，例如 2330.TW）
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
top_n_vol = st.sidebar.slider(
    "挑選高波動資產數量 (Top N)", 1, len(tickers), min(5, len(tickers))
)

rebalance_freq = st.sidebar.selectbox(
    "調倉頻率", ["每月 (Monthly)", "每週 (Weekly)"]
)

run_button = st.sidebar.button("🚀 開始回測", type="primary")

if run_button:
  if len(tickers) < top_n_vol:
    st.error("錯誤：資產池的數量不能小於挑選的高波動資產數量！")
  else:
    with st.spinner("正在下載歷史資料並進行回測運算中..."):
      # 1. 下載資料
      data = yf.download(tickers, start=start_date, end=end_date)[
          "Close"
      ]

      if isinstance(data, pd.Series):
        data = data.to_frame()

      # 移除含有過多空值的欄位
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

      # 過濾在開始日期之後的調倉日
      rebal_dates = [d for d in rebal_dates if d >= data.index[vol_lookback]]

      portfolio_returns = []
      holdings_history = []

      # 4. 回測主迴圈
      for i in range(len(rebal_dates) - 1):
        start_rd = rebal_dates[i]
        end_rd = rebal_dates[i + 1]

        # 取得調倉當日的波動度與 RSI
        if start_rd not in volatility.index:
          # 找最近的有效交易日
          valid_idx = volatility.index.get_indexer(
              [start_rd], method="nearest"
          )[0]
          start_rd = volatility.index[valid_idx]

        current_vol = volatility.loc[start_rd]
        current_rsi = rsi_df.loc[start_rd]

        # 條件篩選：
        # 1. RSI低於指定水準
        # 2. 符合條件中，挑選波動度最高的 Top N
        qualified = current_rsi[current_rsi < rsi_threshold].index
        if len(qualified) > 0:
          # 從符合 RSI 條件的裡面挑選波動度最高的前 N 個
          selected = (
              current_vol[qualified]
              .nlargest(min(top_n_vol, len(qualified)))
              .index.tolist()
          )
        else:
          # 如果沒有任何股票符合 RSI 條件，則退而求其次直接選波動度最高的前 N 個（或空手，此處示範選波動高者）
          selected = current_vol.nlargest(top_n_vol).index.tolist()

        holdings_history.append(
            {"Date": start_rd.strftime("%Y-%m-%d"), "Selected Stocks": selected}
        )

        # 計算這段持有期間內的每日報酬率（等權重配置）
        period_data = daily_returns.loc[start_rd:end_rd, selected].iloc[
            1:
        ]  # 排除調倉當日重複計算
        if not period_data.empty:
          period_portfolio_return = period_data.mean(
              axis=1
          )  # 等權重平均報酬
          portfolio_returns.append(period_portfolio_return)

      if portfolio_returns:
        strategy_rets = pd.concat(portfolio_returns)
        strategy_rets = strategy_rets[
            ~strategy_rets.index.duplicated(keep="first"
        )]

        # 計算基準（大盤，如 SPY 或資產池平均）作對比
        benchmark_rets = daily_returns.mean(axis=1).loc[strategy_rets.index]

        # 累積報酬率
        strategy_cum = (1 + strategy_rets).cumprod()
        benchmark_cum = (1 + benchmark_rets).cumprod()

        # 績效指標計算
        total_return_strat = strategy_cum.iloc[-1] - 1
        total_return_bench = benchmark_cum.iloc[-1] - 1

        annual_return_strat = (
            1 + total_return_strat
        ) ** (252 / len(strategy_rets)) - 1
        annual_vol_strat = strategy_rets.std() * np.sqrt(252)
        sharpe_ratio = (
            annual_return_strat / annual_vol_strat
            if annual_vol_strat != 0
            else 0
        )

        # 最大回撤 (MDD)
        rolling_max = strategy_cum.cummax()
        drawdown = (strategy_cum - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # --- 呈現結果 ---
        st.success("回測完成！")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "策略總報酬率",
            f"{total_return_strat*100:.2f}%",
            f"年化: {annual_return_strat*100:.2f}%",
        )
        col2.metric("策略夏普值 (Sharpe)", f"{sharpe_ratio:.2f}")
        col3.metric("策略最大回撤 (MDD)", f"{max_drawdown*100:.2f}%")
        col4.metric(
            "基準總報酬率 (均值)", f"{total_return_bench*100:.2f}%"
        )

        # 繪製績效曲線圖
        st.subheader("📈 策略 vs 資產池平均 累積績效走勢")
        chart_data = pd.DataFrame(
            {
                "高波動低 RSI 策略": strategy_cum,
                "資產池等權重平均 (Benchmark)": benchmark_cum,
            }
        )
        st.line_chart(chart_data)

        # 顯示最近一次調倉明細
        st.subheader("📋 近期調倉標的紀錄")
        history_df = pd.DataFrame(holdings_history)
        st.dataframe(history_df, use_container_width=True)

      else:
        st.warning(
            "在選定的條件下沒有產生足夠的交易訊號，請嘗試放寬 RSI 閾值或增加資產池。"
        )
