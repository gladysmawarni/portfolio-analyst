import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Financial Performance")

# Header
st.title("📊 Financial Performance")

# Explanation
st.markdown("This tool allows you to upload a CSV of your financial assets and view gains, stock performance, and AI-based recommendations.")

# Upload CSV
st.markdown("### 📁 Browse and Upload CSV")
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

    

if uploaded_file:
    df = pd.read_csv(uploaded_file)

     # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Gains", "Stock Analysis", "Stock Updates", "AI Recommendation"])

    #### ----- Tab 1: Gains ----- ####
    with tab1:
        # Fetch FX rate for USD → EUR
        def get_fx_rate(from_currency: str, to_currency: str = 'EUR') -> float:
            """
            Get live FX rate from Yahoo. Returns 1.0 if from/to are the same.
            """
            if from_currency == to_currency:
                return 1.0
            pair = f"{from_currency}{to_currency}=X"
            try:
                fx_data = yf.Ticker(pair).history(period="1d")
                return fx_data['Close'].iloc[-1]
            except Exception:
                return np.nan
        
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

    
     ## --- Tab 3: Stock Updates ---
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
