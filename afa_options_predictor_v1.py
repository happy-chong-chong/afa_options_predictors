import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================= 1. 核心量化算法 (BS模型对齐口径) =================

# 【高级技巧】：加入缓存装饰器。只要 ticker 和天数不变，拖动滑块时绝不会重复下载，极大提升丝滑度
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hourly_data(ticker, lookback_days):
    """提取过去 N 个交易日的 1 小时级别数据"""
    try:
        fetch_days = int(lookback_days * 1.5) + 10 
        end_date = datetime.now()
        start_date = end_date - timedelta(days=fetch_days)
        
        data = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
        
        if data.empty:
            return None
            
        data['Date'] = data.index.date
        recent_dates = data['Date'].unique()[-lookback_days:]
        data = data[data['Date'].isin(recent_dates)].copy()
        
        # 兼容最新版 yfinance 的多重索引格式
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0]
        else:
            return data['Close'].squeeze()
            
    except Exception as e:
        return None

def calculate_log_returns(prices):
    """计算对数收益率"""
    return np.log(prices / prices.shift(1)).dropna()

def get_market_hours(ticker):
    """智能判断港股和美股的每天交易小时数"""
    if ticker.endswith(".HK"):
        return 5.5 # 港股
    else:
        return 6.5 # 美股

# ================= 2. 网页架构与记忆中枢 =================

st.set_page_config(page_title="群蜂阿发 | 期权价格走势概率预测", layout="wide")
st.title("🐝 群蜂阿发：价格走势概率分布预测 (BS模型底层口径)")

# 初始化记忆中枢：告诉 Streamlit 记住用户的点击状态
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# 侧边栏：参数输入区
st.sidebar.header("📥 第一步：输入基础参数")
ticker = st.sidebar.text_input("股票或指数代码 (如: AAPL, TSLA, 0700.HK)", value="QQQ").upper()
lookback_days = st.sidebar.number_input("分析过去多少个交易日？(默认20天)", min_value=5, max_value=100, value=20, step=1)

# 按下按钮，把记忆中枢设为 True
if st.sidebar.button("🚀 开始提取数据并分析"):
    st.session_state.analyzed = True

# ================= 3. 数据渲染与交互界面 =================

# 只有当记忆中枢为 True 时，才显示下方的所有内容（防止失忆变白）
if st.session_state.analyzed:
    with st.spinner(f"正在分析 {ticker} 过去 {lookback_days} 天的数据..."):
        prices = fetch_hourly_data(ticker, lookback_days)
        
    if prices is None or len(prices) == 0:
        st.error(f"❌ 数据提取失败！请检查：\n1. {ticker} 是否拼写正确（港股必须加 .HK，如 0700.HK）\n2. 网络连接是否正常")
    else:
        # --- 核心计算 ---
        current_price = float(prices.iloc[-1])
        log_returns = calculate_log_returns(prices)
        
        mu_hourly = log_returns.mean()
        sigma_hourly = log_returns.std(ddof=1)
        hours_per_day = get_market_hours(ticker)
        
        st.success(f"✅ 数据拉取成功！当前 {ticker} 最新价: **{current_price:.2f}** | 基于过去 {len(prices)} 个小时 K 线数据。")
        st.write(f"📊 **底层参数**: 小时均值 $\\mu$ = {mu_hourly*100:.4f}%，小时波动率 $\\sigma$ = {sigma_hourly*100:.4f}% (按每日 {hours_per_day} 小时折算)")
        
        st.markdown("### 🎯 第二步：未来 5、10、15 个交易日 90% 概率预测区间")
        
        z_90 = stats.norm.ppf(0.95) 
        cols = st.columns(3)
        horizons = [5, 10, 15]
        
        for i, days in enumerate(horizons):
            target_hours = days * hours_per_day
            mu_target = mu_hourly * target_hours
            sigma_target = sigma_hourly * np.sqrt(target_hours) 
            
            log_upper = mu_target + z_90 * sigma_target
            log_lower = mu_target - z_90 * sigma_target
            
            price_upper = current_price * np.exp(log_upper)
            price_lower = current_price * np.exp(log_lower)
            
            with cols[i]:
                st.info(f"**未来 {days} 个交易日**")
                st.write(f"🔻 下限: **{price_lower:.2f}**")
                st.write(f"🔺 上限: **{price_upper:.2f}**")
                st.caption(f"区间波动率: {sigma_target*100:.2f}%")

        # ================= 4. 互动沙盘：自由滑动图表 =================
        st.markdown("---")
        st.markdown("### 🎛️ 第三步：交互式概率与价格推演图 (对数正态分布)")
        
        target_days_sim = st.slider("选择要推演的未来天数 (交易日)", 1, 30, 5)
        target_hours_sim = target_days_sim * hours_per_day
        mu_sim = mu_hourly * target_hours_sim
        sigma_sim = sigma_hourly * np.sqrt(target_hours_sim)
        
        ctrl_col, chart_col = st.columns([1, 2.5])
        
        with ctrl_col:
            st.subheader("控制台")
            calc_mode = st.radio("你要怎么推演？", ["已知概率 -> 求价格区间", "输入两端价格 -> 求发生概率"])
            
            if calc_mode == "已知概率 -> 求价格区间":
                prob_input = st.slider("设定区间覆盖概率 (%)", 10, 99, 90) / 100.0
                z_sim = stats.norm.ppf(1 - (1 - prob_input) / 2)
                
                sim_log_upper = mu_sim + z_sim * sigma_sim
                sim_log_lower = mu_sim - z_sim * sigma_sim
                
                sim_price_upper = current_price * np.exp(sim_log_upper)
                sim_price_lower = current_price * np.exp(sim_log_lower)
                
                st.write(f"**需要的标准差 (Z值)**: {z_sim:.2f}")
                st.success(f"🔻 **下限**: {sim_price_lower:.2f} \n\n 🔺 **上限**: {sim_price_upper:.2f}")
                
            else: 
                st.caption("提示：在现实世界中，同等金额的涨跌，对应的对数空间并不对称。")
                manual_lower = st.number_input("输入下行止损价 (元)", value=float(current_price * 0.95))
                manual_upper = st.number_input("输入上行止盈价 (元)", value=float(current_price * 1.05))
                
                # 反推并计算概率
                log_ret_lower = np.log(manual_lower / current_price) if manual_lower > 0 else -np.inf
                log_ret_upper = np.log(manual_upper / current_price) if manual_upper > 0 else np.inf
                
                cdf_upper = stats.norm.cdf(log_ret_upper, loc=mu_sim, scale=sigma_sim)
                cdf_lower = stats.norm.cdf(log_ret_lower, loc=mu_sim, scale=sigma_sim)
                total_prob = cdf_upper - cdf_lower
                
                sim_price_lower, sim_price_upper = manual_lower, manual_upper
                st.success(f"股价落在这个区间的总概率: **{total_prob*100:.2f}%**")
                st.write(f"向上突破概率: {(1-cdf_upper)*100:.2f}%")
                st.write(f"向下跌破概率: {cdf_lower*100:.2f}%")
        
        with chart_col:
            # 绘制曲线
            x_returns = np.linspace(mu_sim - 4*sigma_sim, mu_sim + 4*sigma_sim, 500)
            y_pdf = stats.norm.pdf(x_returns, mu_sim, sigma_sim)
            x_prices = current_price * np.exp(x_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_prices, y=y_pdf, mode='lines', name='概率密度', line=dict(color='#007BFF', width=2)))
            
            # 填充选定区间
            fill_mask = (x_prices >= sim_price_lower) & (x_prices <= sim_price_upper)
            fig.add_trace(go.Scatter(
                x=x_prices[fill_mask], y=y_pdf[fill_mask], fill='tozeroy', 
                mode='none', name=f'预测落点概率区域', fillcolor='rgba(0, 123, 255, 0.3)'
            ))
            
            fig.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text="今日收盘价")
            
            fig.update_layout(
                title=f"未来 {target_days_sim} 天 股价概率分布模拟",
                xaxis_title="预测股价",
                yaxis_title="发生概率密度",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)