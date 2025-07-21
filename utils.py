import pandas as pd
import yfinance as yf
import numpy as np

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

## functions

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
            
# --- Fetch historical prices ------------------------------------------
def get_history(ticker, period="1y", interval="1d"):
    """
    Fetch historical price data for a ticker.
    """
    try:
        symbol = ticker
        hist = yf.Ticker(symbol).history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"⚠ Error fetching history for {ticker}: {e}")
        return None
    
# --- Compute moving averages ------------------------------------------
def compute_moving_averages(ticker):
    """
    Fetch price data and compute 50, 100, 200-day moving averages.
    """
    hist = get_history(ticker, period="1y")
    if hist is None or hist.empty:
        print(f"⚠ No price data for {ticker}.")
        return None

    # Calculate moving averages
    hist['MA50']  = hist['Close'].rolling(window=50).mean()
    hist['MA100'] = hist['Close'].rolling(window=100).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()

    # Latest price and moving averages
    latest_price = hist['Close'].iloc[-1]
    latest_ma50  = hist['MA50'].iloc[-1]
    latest_ma100 = hist['MA100'].iloc[-1]
    latest_ma200 = hist['MA200'].iloc[-1]

    st.markdown(f"#### 📊 {ticker} - Moving Averages vs Price")
    st.markdown(f"* Price: €{latest_price:.2f}")
    st.markdown(f"* MA-50: €{latest_ma50:.2f} ({'Above' if latest_price > latest_ma50 else 'Below'})")
    st.markdown(f"* MA-100: €{latest_ma100:.2f} ({'Above' if latest_price > latest_ma100 else 'Below'})")
    st.markdown(f"* MA-200: €{latest_ma200:.2f} ({'Above' if latest_price > latest_ma200 else 'Below'})")

    return hist

## plot moving averages for streamlit
def plot_moving_averages(hist, ticker):
    """
    Interactive Plotly version for Streamlit.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))

    if 'MA50' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], mode='lines', name='50-Day MA', line=dict(dash='dash')))
    if 'MA100' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA100'], mode='lines', name='100-Day MA', line=dict(dash='dot')))
    if 'MA200' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], mode='lines', name='200-Day MA', line=dict(dash='dashdot')))

    fig.update_layout(
        title=f"{ticker} Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (EUR)",
        template="plotly_white",
        height=500,
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig, use_container_width=True)

def compute_volatility(ticker, window=30):
    """
    Calculate rolling volatility (% annualized) for a ticker.
    """
    hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
    if hist.empty:
        print(f"⚠ No data for {ticker}.")
        return None

    # Daily returns
    returns = hist['Close'].pct_change().dropna()

    # Rolling volatility (std dev)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized %

    # Latest volatility value
    latest_vol = rolling_vol.iloc[-1]
    # print(f"📊 {ticker} - {window}-day Volatility: {latest_vol:.2f}%")

    return latest_vol, rolling_vol, hist


def plot_volatility_bar_chart(volatility_results):
    """
    Plots an interactive bar chart of the latest volatilities using Plotly in Streamlit.
    """
    # Convert to DataFrame
    df = pd.DataFrame([
        {"Ticker": t, "Volatility (%)": round(volatility_results[t]["latest_vol"], 2)}
        for t in volatility_results
    ])

    fig = px.bar(df, x="Ticker", y="Volatility (%)",
                #  title="📈 30-Day Rolling Volatility (Annualized)",
                 color_discrete_sequence=["rosybrown"], 
                 text="Volatility (%)"
              )

    fig.update_traces(textposition='outside')  # Position the text above bars

    fig.update_layout(xaxis_title="Ticker", yaxis_title="Volatility (%)",
                      template="plotly_white", height=500)

    st.plotly_chart(fig, use_container_width=True)


def plot_pe_bar_chart(pe_ratios):
    """
    Display a Plotly bar chart of P/E ratios for valid tickers in Streamlit.
    
    Parameters:
        pe_ratios (dict): Dictionary with tickers as keys and P/E ratios as values.
    """
    # Filter out invalid (NaN) P/E values
    valid_pe = {k: round(v,2) for k, v in pe_ratios.items() if not np.isnan(v)}

    if valid_pe:
        df_pe = pd.DataFrame({
            "Ticker": list(valid_pe.keys()),
            "P/E Ratio": list(valid_pe.values())
        })

        fig = px.bar(
            df_pe,
            x="Ticker",
            y="P/E Ratio",
            color_discrete_sequence=["azure"], 
            # title="💰 P/E Ratios Across Portfolio",
            height=500,
            text="P/E Ratio"    
        )

        fig.update_traces(textposition='outside')  # Position the text above bars

        fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="P/E Ratio",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠ No valid P/E ratios to display.")



def plot_beta_bar_chart(beta_values):
    """
    Display a Plotly bar chart of Beta values with a market beta reference line.
    
    Parameters:
        beta_values (dict): Dictionary with tickers as keys and Beta values as floats.
    """
    # Filter out invalid (NaN) Beta values
    valid_beta = {k: v for k, v in beta_values.items() if not np.isnan(v)}

    if valid_beta:
        df_beta = pd.DataFrame({
            "Ticker": list(valid_beta.keys()),
            "Beta": list(valid_beta.values())
        })

        fig = go.Figure()

        # Bars
        fig.add_trace(go.Bar(
            x=df_beta["Ticker"],
            y=df_beta["Beta"],
            marker_color="cornflowerblue",
            name="Beta"
        ))

        # Reference line at Beta = 1
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df_beta["Ticker"]) - 0.5,
            y0=1,
            y1=1,
            line=dict(color="red", dash="dash"),
        )
        fig.add_trace(go.Scatter(
            x=[df_beta["Ticker"].iloc[-1]],
            y=[1],
            mode="text",
            text=["Market Beta (1.0)"],
            textposition="top right",
            showlegend=False
        ))

        fig.update_layout(
            # title="📊 Beta Values Across Portfolio",
            xaxis_title="Ticker",
            yaxis_title="Beta",
            template="plotly_white",
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ No valid Beta values to display.")


def plot_sharpe_ratios(sharpe_ratios):
    """
    Plots an interactive bar chart of Sharpe Ratios using Plotly in Streamlit.
    """
    # Filter out NaNs
    valid_sharpe = {k: round(v,2) for k, v in sharpe_ratios.items() if not np.isnan(v)}

    if valid_sharpe:
        # Convert to DataFrame
        df = pd.DataFrame(list(valid_sharpe.items()), columns=["Ticker", "Sharpe Ratio"])

        # Plotly bar chart
        fig = px.bar(df, x="Ticker", y="Sharpe Ratio",
                    #  title="📈 Sharpe Ratios Across Portfolio",
                     color_discrete_sequence=["lemonchiffon"],
                     text="Sharpe Ratio")

        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        fig.add_hline(y=1, line_dash="dash", line_color="green",
                      annotation_text="Good Sharpe (1.0)", annotation_position="top left")
        fig.add_hline(y=2, line_dash="dash", line_color="blue",
                      annotation_text="Excellent Sharpe (2.0)", annotation_position="top left")

        fig.update_layout(yaxis_title="Sharpe Ratio", xaxis_title="Ticker",
                          template="plotly_white", height=500)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ No valid Sharpe ratios to display.")


def compute_rsi(series, period=14):
    """
    Compute RSI (Relative Strength Index) for a price series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def plot_rsi_values(rsi_values):
    """
    Plots an interactive bar chart of RSI values using Plotly in Streamlit.
    """
    # Filter out NaNs
    valid_rsi = {k: v for k, v in rsi_values.items() if not np.isnan(v)}

    if valid_rsi:
        # Convert to DataFrame
        df = pd.DataFrame(list(valid_rsi.items()), columns=["Ticker", "RSI"])

        # Plotly bar chart
        fig = px.bar(df, x="Ticker", y="RSI",
                    #  title="🔄 RSI (14-day) Across Portfolio",
                     color_discrete_sequence=["skyblue"],
                     text="RSI")

        # Add horizontal lines for Overbought/Oversold thresholds
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Overbought (70)", annotation_position="top left")
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="Oversold (30)", annotation_position="bottom left")

        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

        fig.update_layout(
            yaxis_title="RSI (0-100)",
            xaxis_title="Ticker",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ No valid RSI values to display.")


def compute_macd(series, slow=26, fast=12, signal=9):
    """
    Compute MACD line and Signal line, detect crossover.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Detect latest crossover
    last_macd = macd_line.iloc[-1]
    last_signal = signal_line.iloc[-1]
    crossover = "Bullish" if last_macd > last_signal else "Bearish"

    return last_macd, last_signal, crossover


def plot_macd_crossover(macd_crossovers):
    """
    Plots an interactive bar chart of MACD crossover signals using Plotly in Streamlit.
    """
    # Prepare data
    data = []
    for ticker, val in macd_crossovers.items():
        if val and "Crossover" in val:
            signal = val["Crossover"]
            data.append({
                "Ticker": ticker,
                "Signal": 1 if signal == "Bullish" else -1,
                "Label": signal
            })

    if data:
        df = pd.DataFrame(data)

        # Assign colors based on signal
        df["Color"] = df["Signal"].map({1: "palegreen", -1: "palevioletred"})

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["Ticker"],
            y=df["Signal"],
            marker_color=df["Color"],
            text=df["Label"],
            textposition="outside",
            name="MACD Signal"
        ))

        fig.update_layout(
            # title="📊 MACD Crossover Signals (Latest)",
            yaxis=dict(
                title="Crossover Signal",
                tickmode='array',
                tickvals=[-1, 1],
                ticktext=["Bearish", "Bullish"]
            ),
            xaxis_title="Ticker",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ No MACD crossover signals to display.")