import pandas as pd
import os
import yfinance as yf
from datetime import date
import numpy as np
import streamlit as st
from utils import ( get_fx_rate, compute_moving_averages, plot_moving_averages, compute_volatility, plot_volatility_bar_chart, 
                   plot_pe_bar_chart, plot_beta_bar_chart, plot_sharpe_ratios, compute_rsi, plot_rsi_values, compute_macd,plot_macd_crossover )

st.set_page_config(page_title="Financial Performance")

# Header
st.title("📊 Financial Performance")

# Explanation
st.markdown("This tool allows you to upload a CSV of your financial assets and view gains, stock performance, and AI-based recommendations.")

# Upload CSV
st.markdown("### 📁 Browse and Upload CSV")
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

# Openai Client
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

if uploaded_file:
    df = pd.read_csv(uploaded_file)

     # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Gains", "Stock Analysis", "Stock Updates", "AI Recommendation"])

    #### ----- Tab 1: Gains ----- ####
    with tab1:
        
        # Cache FX rates with EUR = 1.0
        fx_cache = {}
        for ccy in df['Currency Yahoo'].unique():
            if ccy == 'EUR':
                fx_cache[ccy] = 1.0  # no conversion needed
            else:
                fx_cache[ccy] = get_fx_rate(ccy)
            
        def get_price_eur(row) -> float:
            """
            Fetch Yahoo price in native currency and convert to EUR.
            """
            try:
                price_native = (
                    yf.Ticker(row['Ticker'])
                    .history(period="1d")['Close']
                    .iloc[-1]
                )
                fx_rate = fx_cache.get(row['Currency Yahoo'], 1.0)  # default 1.0
                return price_native * fx_rate
            except Exception as e:
                print(f"Error fetching {row['Ticker']}: {e}")   
                return np.nan

        # --- CALCULATIONS ---------------------------------------------
        today = date.today()

        df['Price Today (EUR)']       = df.apply(get_price_eur, axis=1)
        df['Value Today (EUR)']       = df['Units'] * df['Price Today (EUR)']

        df['Gain € Since Last']       = df['Value Today (EUR)'] - df['Value Last Update']
        df['Gain € Since Purchase']   = df['Value Today (EUR)'] - df['Units'] * df['Purchase Price']

        df['Gain % Since Last']       = df['Gain € Since Last']     / df['Value Last Update']                      * 100
        df['Gain % Since Purchase']   = df['Gain € Since Purchase'] / (df['Units'] * df['Purchase Price'])         * 100

        # --- Totals Row -----------------------------------------------
        totals = {
            'Asset'                    : 'TOTAL',
            'Ticker'                   : '',
            'Gain € Since Last'        : df['Gain € Since Last'].sum(),
            'Gain % Since Last'        : df['Gain € Since Last'].sum()     / df['Value Last Update'].sum()           * 100,
            'Gain € Since Purchase'    : df['Gain € Since Purchase'].sum(),
            'Gain % Since Purchase'    : df['Gain € Since Purchase'].sum() / (df['Units']*df['Purchase Price']).sum()* 100,
        }

        # Columns to show
        report_cols = [
            'Asset', 'Ticker',
            'Gain € Since Last', 'Gain % Since Last',
            'Gain € Since Purchase', 'Gain % Since Purchase'
        ]

        report = pd.concat([df[report_cols], pd.DataFrame([totals])], ignore_index=True)

        # --- Display Report in Streamlit ---------------------------------------------
        st.markdown(f"### 📋 Snapshot as of {today} (all converted to EUR)")
        st.dataframe(report.style.format(precision=2))


    ### Tab 2: Stock Analysis
    # --- Subtabs ---
    with tab2:
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7 = st.tabs(["Moving Averages", "Volatility", "P/E Ratio", "Beta", "Sharpe Ratio", "RSI", "MACD"])

        ## --- Moving Averages ---
        with subtab1:
            st.markdown('### 📈 Moving Averages')
            # plot for all tickers in my portfolio
            for ticker in df['Ticker']:
                hist = compute_moving_averages(ticker)
                if hist is not None:
                    plot_moving_averages(hist, ticker)
        
        ## --- Volatility ---
        with subtab2:
            st.markdown('### 📊 30-Day Rolling Volatility (Annualized)')

            # Portfolio tickers
            tickers = df['Ticker'].tolist()

            # Store results
            volatility_results = {}

            for ticker in tickers:
                result = compute_volatility(ticker)
                if result:
                    latest_vol, rolling_vol, hist = result
                    volatility_results[ticker] = {
                        "latest_vol": latest_vol,
                        "rolling_vol": rolling_vol,
                        "price_history": hist
                    }

            plot_volatility_bar_chart(volatility_results) 
        
        ## -- P/E Ratio --
        with subtab3:
            st.markdown('### 💰 P/E Ratios Across Portfolio')

            pe_ratios = {}
            for ticker in df['Ticker']:
                try:
                    info = yf.Ticker(ticker).info
                    pe = info.get('trailingPE', np.nan)

                    if np.isnan(pe):
                        st.warning(f"⚠ {ticker}: No P/E ratio (possibly negative earnings or ETF)")
                    # else:
                    #     st.success(f"📊 {ticker} - P/E Ratio: {pe:.2f}")

                    pe_ratios[ticker] = pe

                except Exception as e:
                    st.error(f"❌ Error fetching P/E for {ticker}: {e}")
                    pe_ratios[ticker] = np.nan
            
            plot_pe_bar_chart(pe_ratios)
       
        ## -- BETA --
        with subtab4:
            st.markdown('### 📊 Beta Values Across Portfolio')

            beta_values = {}
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    beta = info.get('beta', np.nan)

                    if np.isnan(beta):
                        st.warning(f"⚠ {ticker} has no Beta value.")
                    # else:
                    #     st.success(f"📊 {ticker} - Beta: {beta:.2f}")

                    beta_values[ticker] = beta

                except Exception as e:
                    st.error(f"❌ Error fetching Beta for {ticker}: {e}")
                    beta_values[ticker] = np.nan

            plot_beta_bar_chart(beta_values)
        
        ## -- Sharpe Ratio --
        with subtab5:
            st.markdown('### 📊 Sharpe Ratios Across Portfolio')

            sharpe_ratios = {}
            risk_free_rate = 0.04  # Assume 2% annualized risk-free rate

            for ticker in tickers:
                try:
                    # Fetch price history (1 year)
                    hist = yf.Ticker(ticker).history(period="1y")
                    if hist.empty:
                        st.warning(f"⚠ No price data for {ticker}.")
                        sharpe_ratios[ticker] = np.nan
                        continue

                    # Daily returns
                    daily_returns = hist['Close'].pct_change().dropna()

                    # Annualized return & volatility
                    avg_daily_return = daily_returns.mean()
                    annualized_return = avg_daily_return * 252
                    volatility = daily_returns.std() * np.sqrt(252)

                    # Sharpe ratio
                    sharpe = (annualized_return - risk_free_rate) / volatility if volatility != 0 else np.nan

                    sharpe_ratios[ticker] = sharpe
                    # st.success(f"📈 {ticker} - Sharpe Ratio: {sharpe:.2f}")

                except Exception as e:
                    st.error(f"⚠ Error computing Sharpe for {ticker}: {e}")
                    sharpe_ratios[ticker] = np.nan

            plot_sharpe_ratios(sharpe_ratios)

        with subtab6:
            st.markdown('### 🔄 RSI (14-day) Across Portfolio')

            rsi_values = {}

            for ticker in df['Ticker']:
                try:
                    hist = yf.Ticker(ticker).history(period="3mo")
                    if hist.empty:
                        print(f"⚠ No price data for {ticker}.")
                        rsi_values[ticker] = np.nan
                        continue

                    close_prices = hist['Close']
                    rsi_series = compute_rsi(close_prices)
                    latest_rsi = rsi_series.iloc[-1]

                    rsi_values[ticker] = latest_rsi
                    # st.success(f"🔄 {ticker} - RSI: {latest_rsi:.2f}")

                except Exception as e:
                    st.error(f"⚠ Error computing RSI for {ticker}: {e}")
                    rsi_values[ticker] = np.nan

            plot_rsi_values(rsi_values)
        
        with subtab7:
            st.markdown('### 📊 MACD Crossover Signals (Latest)')

            macd_crossovers = {}

            # Loop through portfolio
            for ticker in df['Ticker']:
                try:
                    hist = yf.Ticker(ticker).history(period="3mo")
                    if hist.empty:
                        st.error(f"⚠ No price data for {ticker}.")
                        macd_crossovers[ticker] = None
                        continue

                    close_prices = hist['Close']
                    macd_val, signal_val, crossover = compute_macd(close_prices)

                    macd_crossovers[ticker] = {
                        "MACD": macd_val,
                        "Signal": signal_val,
                        "Crossover": crossover
                    }

                    # st.markdown(f"📈 {ticker} - MACD: {macd_val:.2f}, Signal: {signal_val:.2f}, Crossover: {crossover}")

                except Exception as e:
                    st.error(f"⚠ Error computing MACD for {ticker}: {e}")
                    macd_crossovers[ticker] = None
            
            plot_macd_crossover(macd_crossovers)
    
    ### --- Tab 3: Stock Updates ---
    with tab3:
        # --- Any asset to update ---
        update = st.radio("Do you have any assets to update?", ("No", "Yes"), horizontal=True)

        # if yes
        if update == 'Yes':
            st.markdown("### 🔧 Update Asset Details")
            selected_asset = st.selectbox("Which asset was updated?", df['Asset'].tolist())
            extra_units = st.number_input("How many new units were bought?", min_value=0.0, step=0.1)
            purchase_price = st.number_input("What was the purchase price per unit (EUR)?", min_value=0.0, step=0.01)

            if st.button("Update Asset"):
                if selected_asset and extra_units > 0 and purchase_price > 0:
                    # Find the asset row
                    idx = df[df['Asset'] == selected_asset].index[0]
                    old_units = df.at[idx, 'Units']
                    old_avg_price = df.at[idx, 'Purchase Price']

                    # Recalculate
                    total_cost = (old_units * old_avg_price) + (extra_units * purchase_price)
                    total_units = old_units + extra_units
                    new_avg_price = total_cost / total_units if total_units != 0 else 0

                    # Update DataFrame
                    df.at[idx, 'Units'] = total_units
                    df.at[idx, 'Purchase Price'] = new_avg_price

                    st.success(f"✅ Updated {selected_asset}: {total_units:.4f} units @ avg price €{new_avg_price:.2f}")
                else:
                    st.error("❌ Please fill in all fields with valid values.")


        # --- Any new asset ---
        new_asset = st.radio("Did you add any new assets (Not in current portfolio)?", ("No", "Yes"), horizontal=True)

        # if yes
        if new_asset == 'Yes':
            st.markdown("### 📄 New Asset Details")
            # Input fields for new asset
            asset_name = st.text_input("Asset name")
            ticker = st.text_input("Ticker (Yahoo Finance)")
            units = st.number_input("Units bought", min_value=1.0, step=0.10)
            purchase_price = st.number_input("Purchase price per unit (EUR)", min_value=1.0, step=0.10)
            currency_yahoo = st.selectbox("Currency in Yahoo Finance", ["EUR", "USD"])

            if st.button("Add Asset"):
                if asset_name and ticker and units > 0 and purchase_price > 0:

                    ## check if ticker is valid
                    stock = yf.Ticker(ticker)

                    info = stock.info
                    if "shortName" in info:
                        new_row = {
                        'Asset': asset_name,
                        'Ticker': ticker,
                        'Units': units,
                        'Purchase Price': purchase_price,
                        'Currency Purchase': 'EUR',
                        'Currency Yahoo': currency_yahoo,
                        'Price Last Update': np.nan,
                        'Date Last Update': np.nan,
                        'Value Last Update': np.nan,
                        'Profit Last Update': np.nan
                        }

                        # Append new row to session DataFrame
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])],
                            ignore_index=True
                        )

                        st.success(f"✅ Added {asset_name} to portfolio.")
                    else:
                        st.error("❌ Ticker not found.")

                else:
                    st.error("❌ Please fill in all fields with valid values.")
