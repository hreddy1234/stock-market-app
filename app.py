# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import random
from sklearn.linear_model import LogisticRegression

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Volatility Predictor", layout="wide")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("log.pkl")

model = load_model()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["📄 Description", "📘 Project Details", "📈 Prediction" , "💰 INVESTMENT"]
)

# ===============================
# 📄 DESCRIPTION PAGE
# ===============================
if page == "📄 Description":

    st.title("📊 Volatility Regime swing")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("""
    Stock market is a key platform for investment but is highly volatile and unpredictable.Investors find it difficult to analyze and identify risk levels using traditional methods
    Machine Learning helps analyze historical data and patterns. Machine Learning enables data-driven analysis using historical patterns. This project uses Logistic Regression to classify stocks into HIGH risk and LOW risk
             
    ### 🔹 Objective
    The main objective of this project is to develop a machine learning-based system that can predict the risk level of stocks using historical market data.

    ### 🔹 Output Classes
    - 🟢 LOW Volatility (Stable Market)
    - 🔴 HIGH Volatility (Risky Market)

    ### 🔹 Features Used
    - Stock Prices (OHLC)
    - Volume
    - Daily Returns
    - Volatility (Rolling Std)
    - Moving Average (MA50)
    - RSI Indicator
    - Trend Strength & Drawdown

    ### 🔹 Workflow
    1. Fetch stock data (Yahoo Finance)
    2. Feature Engineering
    3. Model Prediction
    4. Display Result

    👉 Useful for smarter investment decisions.
    """)

# ===============================
# 📘 PROJECT DETAILS
# ===============================
elif page == "📘 Project Details":

    st.title("📘 Project Details")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📊 Data Source")
    st.write("Yahoo Finance API (yFinance)")

    st.subheader("⚙️ Workflow")
    st.markdown("""
    🔽 Data Collection  
    🔽 Data Preprocessing  
    🔽 Feature Engineering  
        • Daily Returns  
        • Volatility (5D)  
        • Moving Average (MA50)  
        • RSI  
        • Drawdown & Trend  

    🔽 Model Training (Logistic Regression)  
    🔽 Prediction (LOW / HIGH Volatility)
    """)

    st.subheader("📈 Output")
    st.write("Predicts volatility.")

# ===============================
# 📈 PREDICTION PAGE
# ===============================
elif page == "📈 Prediction":

    st.title("📈 Stock Volatility Prediction")
    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- INPUT ---------------- #
    ticker = st.selectbox(
        "Select Stock",
        ["hdfc","icici","infy","reliance","sbi","tcs","wipro"]
    )

    # ---------------- BUTTON ---------------- #
    if st.button("Predict Volatility"):

        ticker_map = {
            "hdfc": "HDFCBANK.NS",
            "icici": "ICICIBANK.NS",
            "infy": "INFY.NS",
            "reliance": "RELIANCE.NS",
            "sbi": "SBIN.NS",
            "tcs": "TCS.NS",
            "wipro": "WIPRO.NS"
        }

        stock = ticker_map[ticker]

        with st.spinner("Fetching data..."):

            data = yf.download(stock, period="120d")

            if data.empty or len(data) < 60:
                st.error("❌ Not enough data")
            else:
                # ================= FEATURE ENGINEERING ================= #
                data["return"] = data["Close"].pct_change()
                latest = data.iloc[-1]

                open_price = latest["Open"]
                high = latest["High"]
                low = latest["Low"]
                close_price = latest["Close"]
                volume = latest["Volume"]

                price = close_price

                daily_return = data["return"].iloc[-1]
                volatility_5d = data["return"].rolling(5).std().iloc[-1]
                ma50 = data["Close"].rolling(50).mean().iloc[-1]

                # RSI Calculation
                delta = data["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]

                drawdown = (data["Close"].max() - close_price) / data["Close"].max()
                trend_strength = (close_price - ma50) / ma50
                past_7_days = data["return"].tail(7).sum()

                # ---------------- RETURNS ---------------- #
                weekly_return = float(data['Close'].pct_change(5).iloc[-1])
                monthly_return = float(data['Close'].pct_change(21).iloc[-1])
                quarterly_return = float(data['Close'].pct_change(63).iloc[-1])

                # ================= CLEAN FEATURES ================= #
                raw_features = [
                    open_price, close_price, low, high, close_price, volume, 15,
                    daily_return, volatility_5d, ma50, rsi,
                    drawdown, trend_strength, past_7_days
                ]

                features = []
                for x in raw_features:
                    try:
                        val = float(x)
                        if np.isnan(val) or np.isinf(val):
                            val = 0
                    except:
                        val = 0
                    features.append(val)

                # ONE HOT ENCODING
                tickers = ['hdfc','icici','infy','reliance','sbi','tcs','wipro']
                encoded = [1 if ticker == t else 0 for t in tickers]

                columns = [
                    'open_price','price','low','high','close_price','volume','vix',
                    'daily_return','volatility_5d','ma50','rsi',
                    'drawdown','trend_strength','past_7_days',
                    'ticker_hdfc','ticker_icicibank','ticker_infy',
                    'ticker_reliance','ticker_sbi','ticker_tcs','ticker_wipro'
                ]

                row = pd.DataFrame([features + encoded], columns=columns)

                # ---------------- PREDICTION ---------------- #
                probs = model.predict_proba(row)[0]
                classes = model.classes_

                prob_dict = dict(zip(classes, probs))

                prob_low = prob_dict.get("LOW", 0)
                prob_high = prob_dict.get("HIGH", 0)

                confidence = max(prob_low, prob_high)

                if confidence > 0.6:
                    result = "HIGH" if prob_high > prob_low else "LOW"
                else:
                    result = "HIGH" if volatility_5d > 0.02 else "LOW"

                # ---------------- TREND + RECOMMENDATION ---------------- #
                trend = "UPTREND" if float(close_price) > float(ma50) else "DOWNTREND"

                if result == "LOW" and trend == "UPTREND" and 30 < rsi < 70:
                    recommendation = "🟢 BUY:👉The market is stable with low volatility, the stock is trending upward, and momentum is healthy. This is a good opportunity to enter and invest."
                elif result == "HIGH":
                    recommendation = "🔴 AVOID👉The market is highly volatile and risky right now. Price movements are unpredictable, so it's safer to stay out and protect your capital."
                else:
                    recommendation = "🟡 HOLD👉The situation is unclear or not strong enough for buying. If you already own the stock, keep holding and wait for a better signal."
                
                # ---------------- STORE DATA FOR INVESTMENT PAGE ---------------- #
                st.session_state["ticker"] = ticker
                st.session_state["result"] = result
                st.session_state["trend"] = trend
                st.session_state["price"] = float(price)

                st.session_state["weekly_return"] = weekly_return
                st.session_state["monthly_return"] = monthly_return
                st.session_state["quarterly_return"] = quarterly_return

                st.session_state["recommendation"] = recommendation
                                
                st.subheader("📊 Key Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:          
                    st.metric("Ticker", ticker.upper())
                    st.metric("low Price", f"{float(price):<20.2f}")
                    st.metric("Open Price", f"{float(open_price):<20.2f}")
                    st.metric("High Price", f"{float(high):<20.2f}")

                with col2:
                    st.metric("current Price", f"{float(low):<20.2f}")
                    st.metric("Volume", f"{int(volume):<20,}")
                    st.metric("Daily Return", f"{daily_return:<20.5f}")
                    st.metric("Volatility (5D)", f"{volatility_5d:<20.5f}")

                with col3:
                    st.metric("MA50", f"{float(ma50):<20.2f}")
                    st.metric("RSI", f"{float(rsi):<20.2f}")
                    st.metric("Past 7 Days", f"{past_7_days:<20.5f}")
                st.markdown("---")
                
                st.markdown("### 📊 Prediction Result")

                if result == "HIGH":
                    st.error(f"🔴 HIGH Volatility")
                    st.warning("⚠️ The market is currently experiencing high volatility. This stock appears unstable, with prices likely to fluctuate significantly. Investors are advised to proceed with caution.")
                else:
                    st.success(f"🟢 LOW Volatility")
                    st.info("✅ The market is currently exhibiting stable conditions. This stock shows relatively low volatility, with smoother and more predictable price movements. It may be considered a lower-risk option under current market conditions.")
                st.markdown("---")
                 # ---------------- RECOMMENDATION ---------------- #
                
                if "BUY" in recommendation:
                    st.success(recommendation)

                elif "AVOID" in recommendation or "SELL" in recommendation:
                    st.error(recommendation)

                else:
                    st.warning(recommendation)

                if weekly_return < 0:
                    st.error("\n⚠️ Market showing negative returns. Avoid investing.")
                st.success("✅ Prediction saved! Go to 💰 Investment tab")
                st.session_state["show_invest"] = True
# ===============================
# 💰 INVESTMENT 
# ===============================
elif page == "💰 INVESTMENT":

    # ===============================
    # INVESTMENT SECTION
    # ===============================
    if st.session_state.get("show_invest", False):

        st.title("💰 Investment Decision")
        st.markdown("<hr>", unsafe_allow_html=True)

        # ===============================
        # 📊 SHOW PREDICTION DATA
        # ===============================
        if "result" not in st.session_state:
            st.error("⚠️ No prediction data found. Please run prediction first.")
            st.stop()

        st.subheader("📊 Prediction Summary")
        st.markdown("---")

        ticker = st.session_state["ticker"]
        result = st.session_state["result"]
        trend = st.session_state["trend"]
        price = st.session_state["price"]

        weekly_return = st.session_state["weekly_return"]
        monthly_return = st.session_state["monthly_return"]
        quarterly_return = st.session_state["quarterly_return"]

        col1, col2, col3 = st.columns(3)

        col1.metric("📌 Ticker", ticker.upper())
        col1.metric("💰 Price", f"₹{price:.2f}")

        col2.metric("📈 Weekly Return", f"{weekly_return*100:.2f}%")
        col2.metric("📊 Monthly Return", f"{monthly_return*100:.2f}%")

        col3.metric("📉 Quarterly Return", f"{quarterly_return*100:.2f}%")
        col3.metric("📊 Trend", trend)

        # RESULT
        if result == "HIGH":
            st.error("🔴 HIGH Volatility")
        else:
            st.success("🟢 LOW Volatility")

        st.markdown("---")

        # ---------------- INFO ---------------- #
        st.info(f"📌 Using prediction for: {ticker.upper()}")

        # ---------------- USER DECISION ---------------- #
        choice = st.radio(
            "Do you want to invest based on this prediction?",
            ["Yes", "No"]
        )

        if choice == "No":
            st.info("👍 No investment made. Exiting...")
            st.stop()

        # ---------------- INVESTMENT INPUT ---------------- #
        base_capital = st.number_input(
            "Enter your investment amount (₹):",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

        if base_capital == 0:
            st.error("❌ Please enter some investment amount greater than ₹0")
            st.stop()

        # ===============================
        # BUTTON → CALCULATE INVESTMENT
        # ===============================
        if st.button("🔮 Predict & Show Investment Plan"):

            # ---------------- STRATEGY ---------------- #
            if result == "LOW" and trend == "UPTREND" and weekly_return > 0:
                invest_ratio = 0.6
                strategy_msg = "🟢 Strong opportunity: Uptrend + Low volatility"

            elif result == "LOW":
                invest_ratio = 0.4
                strategy_msg = "🟡 Moderate opportunity: Low volatility"

            elif result == "HIGH":
                invest_ratio = 0.2
                strategy_msg = "🔴 Risky market: High volatility"

            else:
                invest_ratio = 0.3
                strategy_msg = "🟠 Neutral market condition"

            # ---------------- CALCULATIONS ---------------- #
            suggested_investment = base_capital * invest_ratio
            shares = suggested_investment / float(price)

            weekly_profit = suggested_investment * weekly_return
            monthly_profit = suggested_investment * monthly_return
            quarterly_profit = suggested_investment * quarterly_return

            # ===============================
            # DISPLAY
            # ===============================
            st.markdown("## 💰 Investment Plan")
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("💼 Capital", f"₹{base_capital:<23,.0f}")
            col2.metric("📊 Invest %", f"{invest_ratio*100:<25.0f}%")
            col3.metric("💰 Invested", f"₹{suggested_investment:<23,.0f}")
            col4.metric("📈 Shares", f"{shares:<25.2f}")

            st.markdown("---")

            # ---------------- RETURNS ---------------- #
            st.subheader("📊 Expected Returns (%)")

            col1, col2, col3 = st.columns(3)

            col1.metric("1 Week", f"{weekly_return*100:<25.2f}%")
            col2.metric("1 Month", f"{monthly_return*100:<25.2f}%")
            col3.metric("1 Quarter", f"{quarterly_return*100:<25.2f}%")

            st.markdown("---")

            # ---------------- PROFIT ---------------- #
            st.subheader("💸 Estimated Profit (₹)")

            col1, col2, col3 = st.columns(3)

            col1.metric("1 Week", f"₹{weekly_profit:<25.2f}")
            col2.metric("1 Month", f"₹{monthly_profit:<25.2f}")
            col3.metric("1 Quarter", f"₹{quarterly_profit:<25.2f}")

            st.markdown("---")

            # ---------------- SUGGESTIONS ---------------- #
            st.subheader("🧠 Investment Suggestions")

            if result == "HIGH":
                if trend == "DOWNTREND":
                    st.error("📉 High Volatility + Downtrend → Very risky market.")
                else:
                    st.warning("⚠️ High Volatility → Invest small amount with stop-loss.")

            elif result == "LOW":
                if trend == "UPTREND":
                    st.success("📈 Low Volatility + Uptrend → Ideal condition.")
                else:
                    st.info("📊 Low Volatility but weak trend → Moderate investment.")

            else:
                st.warning("⚠️ Uncertain conditions → Wait and watch.")

            # ---------------- TARGET / STOP LOSS ---------------- #
            target_price = price * (1 + abs(monthly_return))
            stop_loss = price * (1 - abs(weekly_return))

            col1, col2 = st.columns(2)

            col1.success(f"🎯 Target Price: ₹{target_price:<25.2f}")
            col2.error(f"🛑 Stop Loss: ₹{stop_loss:<25.2f}")

            # ---------------- GRAPH ---------------- #
            import numpy as np
            import pandas as pd

            days = 30
            growth = [base_capital]

            for i in range(1, days):
                noise = np.random.normal(0, 0.003)
                growth.append(growth[-1] * (1 + (weekly_return/5) + noise))

            df = pd.DataFrame({
                "Day": range(days),
                "Portfolio Value": growth
            })

            st.subheader("📈 Smart Portfolio Growth")
            st.line_chart(df.set_index("Day"))