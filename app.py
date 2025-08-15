import pandas as pd
import os
import yfinance as yf
from datetime import date
import numpy as np
import streamlit as st
from langchain_openai import ChatOpenAI
from utils import ( get_fx_rate, compute_moving_averages, plot_moving_averages, compute_volatility, plot_volatility_bar_chart, 
                   plot_pe_bar_chart, plot_beta_bar_chart, plot_sharpe_ratios, compute_rsi, plot_rsi_values, compute_macd,plot_macd_crossover ) # custom functions

# Tab title
st.set_page_config(page_title="Portfolio Analyst")

# Header
st.title("📊 Financial Performance")

# Explanation
st.markdown("This tool allows you to upload a CSV of your financial assets and view gains, stock performance, and AI-based recommendations.")

# Upload CSV
st.markdown("### 📁 Browse and Upload CSV")
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

# Openai Client
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

## -- Session State -- ##
# to check if the data is uploaded
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False  

# save uploaded df and continue to the analysis
if uploaded_file and st.session_state.data_loaded == False:
    st.session_state.df = pd.read_csv(uploaded_file) 
    st.session_state.data_loaded = True

#  Flag to start analysing, after the user specify new assets
if 'flag' not in st.session_state:
    st.session_state.flag = False  

## -- Start of Analysis -- ##
if  st.session_state.data_loaded == True:
    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Gains", "Stock Updates", "Export Data",  "Stock Analysis", "AI Recommendation", "Potential Investments"])

    #### ----- Tab 1: Gains ----- ####
    with tab1:
        
        # Cache FX rates with EUR = 1.0
        fx_cache = {}
        for ccy in st.session_state.df['Currency Yahoo'].unique():
            if ccy == 'EUR':
                fx_cache[ccy] = 1.0  # no conversion needed
            else:
                fx_cache[ccy] = get_fx_rate(ccy) # Fetch and cache FX rate for other currencies
        
        # Define a function to get the latest price in EUR
        def get_price_eur(row) -> float:
            """
            Fetch Yahoo price in native currency and convert to EUR.
            """
            try:
                # Get the most recent closing price from Yahoo Finance
                price_native = (
                    yf.Ticker(row['Ticker'])
                    .history(period="1d")['Close']
                    .iloc[-1]
                )
                 # Get the FX rate from the cache (default to 1.0 if missing)
                fx_rate = fx_cache.get(row['Currency Yahoo'], 1.0) 
                # Convert the price to EUR
                return price_native * fx_rate
            except Exception as e:
                st.error(f"Error fetching {row['Ticker']}: {e}")   
                return np.nan


        # --- CALCULATIONS ---------------------------------------------
        today = date.today()

        st.session_state.df['Price Today (EUR)']       = st.session_state.df.apply(get_price_eur, axis=1)
        st.session_state.df['Value Today (EUR)']       = st.session_state.df['Units'] * st.session_state.df['Price Today (EUR)']

        st.session_state.df['Gain € Since Last']       = st.session_state.df['Value Today (EUR)'] - st.session_state.df['Value Last Update']
        st.session_state.df['Gain € Since Purchase']   = st.session_state.df['Value Today (EUR)'] - st.session_state.df['Units'] * st.session_state.df['Purchase Price']

        st.session_state.df['Gain % Since Last']       = st.session_state.df['Gain € Since Last']     / st.session_state.df['Value Last Update']                      * 100
        st.session_state.df['Gain % Since Purchase']   = st.session_state.df['Gain € Since Purchase'] / (st.session_state.df['Units'] * st.session_state.df['Purchase Price'])         * 100

        # --- Totals Row -----------------------------------------------
        totals = {
            'Asset'                    : 'TOTAL',
            'Ticker'                   : '',
            'Gain € Since Last'        : st.session_state.df['Gain € Since Last'].sum(),
            'Gain % Since Last'        : st.session_state.df['Gain € Since Last'].sum()     / st.session_state.df['Value Last Update'].sum()           * 100,
            'Gain € Since Purchase'    : st.session_state.df['Gain € Since Purchase'].sum(),
            'Gain % Since Purchase'    : st.session_state.df['Gain € Since Purchase'].sum() / (st.session_state.df['Units']*st.session_state.df['Purchase Price']).sum()* 100,
        }

        # Columns to show
        report_cols = [
            'Asset', 'Ticker',
            'Gain € Since Last', 'Gain % Since Last',
            'Gain € Since Purchase', 'Gain % Since Purchase'
        ]

        report = pd.concat([st.session_state.df[report_cols], pd.DataFrame([totals])], ignore_index=True)

        # --- Display Report in Streamlit ---------------------------------------------
        st.markdown(f"### 📋 Snapshot as of {today} (all converted to EUR)")
        st.dataframe(report.style.format(precision=2))


    #### ----- Tab 2: Asset Updates ----- ####
    with tab2:
        # --- Any asset to update ---
        update = st.radio("Do you have any assets to update?", ("No", "Yes"), horizontal=True)

        # if yes
        if update == 'Yes':
            st.markdown("### 🔧 Update Asset Details")
            selected_asset = st.selectbox("Which asset was updated?", st.session_state.df['Asset'].tolist())
            extra_units = st.number_input("How many new units were bought?", min_value=0.0, step=0.1)
            purchase_price = st.number_input("What was the purchase price per unit (EUR)?", min_value=0.0, step=0.01)

            if st.button("Update Asset"):
                if selected_asset and extra_units > 0 and purchase_price > 0:
                    # Find the asset row
                    idx = st.session_state.df[st.session_state.df['Asset'] == selected_asset].index[0]
                    old_units = st.session_state.df.at[idx, 'Units']
                    old_avg_price = st.session_state.df.at[idx, 'Purchase Price']

                    # Recalculate
                    total_cost = (old_units * old_avg_price) + (extra_units * purchase_price)
                    total_units = old_units + extra_units
                    new_avg_price = total_cost / total_units if total_units != 0 else 0

                    # Update DataFrame
                    st.session_state.df.at[idx, 'Units'] = total_units
                    st.session_state.df.at[idx, 'Purchase Price'] = new_avg_price

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
                        st.session_state.df = pd.concat(
                            [st.session_state.df, pd.DataFrame([new_row])],
                            ignore_index=True
                        )  


                        st.success(f"✅ Added {asset_name} to portfolio.")
                    else:
                        st.error("❌ Ticker not found.")

                else:
                    st.error("❌ Please fill in all fields with valid values.")

        # if the analyze button is clicked, we start the stock analysis
        if st.button('Analyze'):
            st.session_state.flag = True


    #### ----- Tab 3: Export Data ----- ####
    with tab3:
        if st.session_state.flag == True:
            selected_df = st.session_state.df[['Asset', 'Ticker', 'Units', 'Purchase Price', 'Currency Purchase', 'Currency Yahoo', 'Price Last Update',
                                            'Date Last Update', 'Value Last Update', 'Profit Last Update']]
            

            ## -- Update Data --
            # current price (EUR)
            selected_df['Price Last Update'] = selected_df.apply(get_price_eur, axis=1)

            # current date
            today = date.today().strftime('%Y-%m-%d')
            selected_df['Date Last Update'] = today

            # current value
            selected_df['Value Last Update'] = selected_df['Units'] * selected_df['Price Last Update']

            # current profit (current value - original value)
            selected_df['original_value'] = selected_df['Units'] * selected_df['Purchase Price']
            selected_df['Profit Last Update'] = selected_df['Value Last Update'] - selected_df['original_value']

            selected_df.drop(columns=['original_value'], inplace=True)

            # show df
            st.dataframe(selected_df)

            # Convert DataFrame to CSV
            csv = selected_df.to_csv(index=False).encode('utf-8')

            # Download button
            st.download_button(
                label="📥 Download Updated Assets",
                data=csv,
                file_name=f"assets {today}.csv",
                mime='text/csv'
            )

            st.session_state.flag = True


    #### ----- Tab 4: Stock Analysis ----- ####
    with tab4:
        # --- Subtabs ---
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7 = st.tabs(["Moving Averages", "Volatility", "P/E Ratio", "Beta", "Sharpe Ratio", "RSI", "MACD"])

        if st.session_state.flag == True:
            ## --- Moving Averages ---
            with subtab1:
                st.markdown('### 📈 Moving Averages')
                # plot for all tickers in my portfolio
                for ticker in st.session_state.df['Ticker']:
                    hist = compute_moving_averages(ticker, True)
                    if hist is not None:
                         # show charts
                        plot_moving_averages(hist, ticker)
            
            ## --- Volatility ---
            with subtab2:
                st.markdown('### 📊 30-Day Rolling Volatility (Annualized)')

                # Portfolio tickers
                tickers = st.session_state.df['Ticker'].tolist()

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

                # show charts
                plot_volatility_bar_chart(volatility_results) 
                
            
            ## --- P/E Ratio ---
            with subtab3:
                st.markdown('### 💰 P/E Ratios Across Portfolio')

                pe_ratios = {}
                for ticker in st.session_state.df['Ticker']:
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
                
                # show charts
                plot_pe_bar_chart(pe_ratios)
        
            ## --- BETA ---
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

                # show charts
                plot_beta_bar_chart(beta_values)
            

            ## --- Sharpe Ratio ---
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

                # show charts
                plot_sharpe_ratios(sharpe_ratios)


            ## --- RSI ---
            with subtab6:
                st.markdown('### 🔄 RSI (14-day) Across Portfolio')

                rsi_values = {}

                for ticker in st.session_state.df['Ticker']:
                    try:
                        hist = yf.Ticker(ticker).history(period="3mo")
                        if hist.empty:
                            st.warn(f"⚠ No price data for {ticker}.")
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

                # show charts
                plot_rsi_values(rsi_values)
        

            ## --- MACD ---
            with subtab7:
                st.markdown('### 📊 MACD Crossover Signals (Latest)')

                macd_crossovers = {}

                # Loop through portfolio
                for ticker in st.session_state.df['Ticker']:
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
                
                # show charts
                plot_macd_crossover(macd_crossovers)
        
    
    #### ----- Tab 5: AI Recommendations ----- ####
    with tab5:
        if st.session_state.flag == True:
            st.markdown("### 🤖 Personalized AI Recommendation")
            with st.spinner('Analyzing...'):

                llm = ChatOpenAI(
                    model="gpt-5-nano",
                    temperature=0)
                
                # Combine all KPI dictionaries into a single DataFrame
                kpi_df = pd.DataFrame({
                    'Ticker': st.session_state.df['Ticker'],
                    'Asset': st.session_state.df['Asset'],
                    'Price Today': st.session_state.df['Price Today (EUR)'],
                    'MA-50': [compute_moving_averages(t, False)['Close'].rolling(50).mean().iloc[-1] for t in st.session_state.df['Ticker']],
                    'MA-100': [compute_moving_averages(t,False)['Close'].rolling(100).mean().iloc[-1] for t in st.session_state.df['Ticker']],
                    'MA-200': [compute_moving_averages(t,False)['Close'].rolling(200).mean().iloc[-1] for t in st.session_state.df['Ticker']],
                    'Volatility (30d)': [volatility_results[t]['latest_vol'] for t in st.session_state.df['Ticker']],
                    'P/E Ratio': [pe_ratios.get(t, np.nan) for t in st.session_state.df['Ticker']],
                    'Beta': [beta_values.get(t, np.nan) for t in st.session_state.df['Ticker']],
                    'Sharpe Ratio': [sharpe_ratios.get(t, np.nan) for t in st.session_state.df['Ticker']],
                    'RSI': [rsi_values.get(t, np.nan) for t in st.session_state.df['Ticker']],
                    'MACD Crossover': [macd_crossovers.get(t, {}).get('Crossover', 'N/A') for t in st.session_state.df['Ticker']]
                })
                
                # Analysis with GPT
                messages = [
                    (
                        "system",
                        """
                        You are a portfolio analyst.

                        For every ticker:
                        • Summarise key strengths & risks based on MA vs price, volatility, P/E, Beta, Sharpe, RSI, MACD.
                        • Flag momentum signals: RSI (>70 overbought, <30 oversold) and MACD crossovers.
                        • Recommend: 'Buy', 'Hold', or 'Reduce', with 1–2 line rationale.

                        Finish with a brief overall portfolio note.
                        Return the answer in a markdown format, without specifying it is markdown.
                        """,
                    ),
                    ("human", kpi_df.to_string()
                ),
                ]

                ai_msg = llm.invoke(messages)
            
             # show the KPI
            st.write('Portfolio Summary')
            st.dataframe(kpi_df)
            # show analysis
            st.markdown(ai_msg.content)


    #### ----- Tab 6: New Potential Stock Analysis  ----- ####
    with tab6:
        if st.session_state.flag == True:
            # --- New tickers to analyze ----------------------------------------
            text_input = st.text_input(
                    label="🆕 Enter Ticker Symbols for New Stocks",
                    placeholder="e.g. NVDA, TSLA, AAPL (separate with commas)",
                    label_visibility='visible'
                )

            if st.button('Submit') or text_input.strip():
                with st.spinner('Analyzing...'):
                    # split input by commas
                    new_tickers = [ticker.strip() for ticker in text_input.split(',')]
                    # list to save not valid tickers
                    not_valid_tickers = []

                    # Create placeholder dictionaries for KPI results
                    new_volatility_results = {}
                    new_pe_ratios = {}
                    new_beta_values = {}
                    new_sharpe_ratios = {}
                    new_rsi_values = {}
                    new_macd_crossovers = {}

                    # --- Compute KPIs for each new ticker -----------------------------
                    for ticker in new_tickers:
                        ## check if ticker is valid
                        stock = yf.Ticker(ticker)

                        info = stock.info

                        if "regularMarketDayRange" in info:
                            # Volatility
                            result = compute_volatility(ticker)
                            if result:
                                latest_vol, rolling_vol, hist = result
                                new_volatility_results[ticker] = {
                                    "latest_vol": latest_vol,
                                    "rolling_vol": rolling_vol,
                                    "price_history": hist
                                }

                            # P/E Ratio
                            try:
                                info = yf.Ticker(ticker).info
                                pe = info.get('trailingPE', np.nan)
                                new_pe_ratios[ticker] = pe
                            except Exception:
                                new_pe_ratios[ticker] = np.nan

                            # Beta
                            try:
                                beta = info.get('beta', np.nan)
                                new_beta_values[ticker] = beta
                            except Exception:
                                new_beta_values[ticker] = np.nan

                            # Sharpe Ratio
                            try:
                                hist = yf.Ticker(ticker).history(period="1y")
                                daily_returns = hist['Close'].pct_change().dropna()
                                avg_daily_return = daily_returns.mean()
                                annualized_return = avg_daily_return * 252
                                volatility = daily_returns.std() * np.sqrt(252)
                                sharpe = (annualized_return - risk_free_rate) / volatility if volatility != 0 else np.nan
                                new_sharpe_ratios[ticker] = sharpe
                            except Exception:
                                new_sharpe_ratios[ticker] = np.nan

                            # RSI
                            try:
                                close_prices = hist['Close']
                                rsi_series = compute_rsi(close_prices)
                                latest_rsi = rsi_series.iloc[-1]
                                new_rsi_values[ticker] = latest_rsi
                            except Exception:
                                new_rsi_values[ticker] = np.nan

                            # MACD
                            try:
                                macd_val, signal_val, crossover = compute_macd(close_prices)
                                new_macd_crossovers[ticker] = {
                                    "MACD": macd_val,
                                    "Signal": signal_val,
                                    "Crossover": crossover
                                }
                            except Exception:
                                new_macd_crossovers[ticker] = None
                        
                        else:
                            st.error(f'❌ No information for ticker: {ticker}.')
                            not_valid_tickers.append(ticker)
                            

                    # valid tickers
                    new_tickers = [x for x in new_tickers if x not in not_valid_tickers]
                    st.write(f"Analyzing the following assets: {', '.join(new_tickers)}")

                    # Combine your portfolio and new tickers into one KPI DataFrame
                    combined_kpi_df = pd.DataFrame({
                        'Ticker': list(st.session_state.df['Ticker']) + new_tickers,
                        'Asset': list(st.session_state.df['Asset']) + new_tickers,  # using ticker as asset name for new ones
                        'Price Today': list(st.session_state.df['Price Today (EUR)']) + [
                            yf.Ticker(t).history(period="1d")['Close'].iloc[-1] for t in new_tickers
                        ],
                        'Volatility (30d)': [volatility_results[t]['latest_vol'] for t in st.session_state.df['Ticker']] +
                                            [new_volatility_results[t]['latest_vol'] for t in new_tickers],
                        'P/E Ratio': [pe_ratios.get(t, np.nan) for t in st.session_state.df['Ticker']] +
                                    [new_pe_ratios.get(t, np.nan) for t in new_tickers],
                        'Beta': [beta_values.get(t, np.nan) for t in st.session_state.df['Ticker']] +
                                [new_beta_values.get(t, np.nan) for t in new_tickers],
                        'Sharpe Ratio': [sharpe_ratios.get(t, np.nan) for t in st.session_state.df['Ticker']] +
                                        [new_sharpe_ratios.get(t, np.nan) for t in new_tickers],
                        'RSI': [rsi_values.get(t, np.nan) for t in st.session_state.df['Ticker']] +
                            [new_rsi_values.get(t, np.nan) for t in new_tickers],
                        'MACD Crossover': [macd_crossovers.get(t, {}).get('Crossover', 'N/A') for t in st.session_state.df['Ticker']] +
                                        [new_macd_crossovers.get(t, {}).get('Crossover', 'N/A') for t in new_tickers]
                    })


                    # Analysis by GPT
                    adjustment_instruction = f"""
                    You are a portfolio strategist.

                    1. Review my current portfolio KPIs and the additional stocks analyzed.
                    2. Suggest portfolio adjustments:
                    • Which assets to increase, reduce, or remove.
                    • Whether to include any of the new tickers ({', '.join(new_tickers)}).
                    • Keep recommendations balanced between risk and return.

                    Respond in markdown format with clear bullet points.
                    """

                    messages = [
                        ("system", adjustment_instruction),
                        ("human", combined_kpi_df.to_string())
                    ]

                    # Get GPT recommendation
                    ai_response = llm.invoke(messages)
                
                # show analysis
                st.markdown(ai_response.content)