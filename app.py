```python
import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time
import zipfile
from io import BytesIO

# Override pandas_datareader with yfinance for compatibility
yf.pdr_override()

# Page config
st.set_page_config(page_title="Supply Demand Screener", layout="wide")

st.title("Supply Demand Stock Screener")

# Sidebar for inputs
st.sidebar.header("Screener Settings")

# 1. Script Type Selection
script_type = st.sidebar.selectbox(
    "Select Script Type",
    options=[
        "FNO NIFTY50 ALL STOCK",
        "TEST SMALL LIST"  # Simplified for debugging
    ],
    index=1  # Default to TEST SMALL LIST
)

# Symbol lists
symbol_lists = {
    "FNO NIFTY50 ALL STOCK": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS",
        "HCLTECH.NS", "ITC.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "TITAN.NS", "NESTLEIND.NS",
        "BAJAJFINSV.NS", "POWERGRID.NS", "M&M.NS", "NTPC.NS", "TECHM.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
        "GRASIM.NS", "DIVISLAB.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "ADANIPORTS.NS",
        "CIPLA.NS", "BPCL.NS", "SHREECEM.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
        "TATACONSUM.NS", "UPL.NS", "COALINDIA.NS", "ONGC.NS", "HINDALCO.NS", "INDUSINDBK.NS",
        "ADANIENT.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS"
    ],
    "TEST SMALL LIST": ["RELIANCE.NS", "TCS.NS", "SBIN.NS", "ITC.NS"]
}
symbols = symbol_lists.get(script_type, ["RELIANCE.NS"])
custom_symbols = st.sidebar.text_area("Or Enter Custom Symbols (comma-separated)", value=",".join(symbols))
if custom_symbols:
    symbols = [s.strip() for s in custom_symbols.split(",") if s.strip()]

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", datetime.now())

# 2. Number of Base Candles
num_base_candles = st.sidebar.slider("Number of Base Candles", min_value=1, max_value=6, value=1)

# 3. Time Interval
intervals = {
    "1 Min": "1m",
    "5 Min": "5m",
    "15 Min": "15m",
    "30 Min": "30m",
    "60 Min": "60m",
    "1 HR": "1h",
    "DAILY": "1d",
    "WEEKLY": "1wk"
}
display_intervals = list(intervals.keys())
selected_interval_display = st.sidebar.selectbox("Select Time Interval", options=display_intervals, index=6)  # Default to DAILY
yf_interval = intervals[selected_interval_display]

# 4. Zone Status
zone_status = st.sidebar.selectbox("Zone Status", options=["ALL", "FRESH", "TARGET", "STOPLOSS"], index=0)

# 5. Zone Type
zone_type = st.sidebar.selectbox("Zone Type", options=["ALL", "SUPPLY", "DEMAND"], index=0)

# Pattern Settings
st.sidebar.header("Pattern Settings")
min_body_rally_drop = st.sidebar.slider("Min Body % for Rally/Drop", 0, 100, 80)
max_body_base = st.sidebar.slider("Max Body % for Base", 0, 100, 50)

# Risk-Reward Settings
st.sidebar.header("Entry & Stoploss Settings")
rr_ratio = 5.0  # Fixed RR 1:5
sl_buffer_pct = st.sidebar.number_input("Stoploss Buffer %", min_value=0.0, value=1.0, step=0.5)

# CSV Upload Fallback
st.sidebar.header("Manual Data Upload (Fallback)")
uploaded_file = st.sidebar.file_uploader("Upload CSV (OHLC data)", type=["csv"])

# Fetch data function
@st.cache_data
def fetch_data(symbols, start, end, interval):
    data = {}
    max_attempts = 3
    for symbol in symbols:
        attempt = 1
        success = False
        temp_start = start
        temp_end = end
        while attempt <= max_attempts:
            try:
                df = pdr.get_data_yahoo(symbol, start=temp_start, end=temp_end, interval=interval)
                if not df.empty and all(col in df.columns for col in ['High', 'Low', 'Close', 'Open']):
                    df = df.dropna(subset=['High', 'Low', 'Close', 'Open'])
                    df[['High', 'Low', 'Close', 'Open']] = df[['High', 'Low', 'Close', 'Open']].astype(float)
                    if not df.empty:
                        data[symbol] = df
                        st.info(f"Success: Fetched data for {symbol} ({len(df)} rows, interval: {interval}, {temp_start} to {temp_end}).")
                        success = True
                        break
                    else:
                        st.warning(f"No valid data for {symbol} after cleaning (all rows dropped due to NaN).")
                else:
                    st.warning(f"No valid data for {symbol} (empty or missing columns: {df.columns.tolist() if not df.empty else 'Empty'}).")
            except Exception as e:
                st.error(f"Error fetching {symbol} (attempt {attempt}/{max_attempts}): {str(e)}")
            attempt += 1
            if attempt <= max_attempts:
                time.sleep(1)
                if interval in ['1m', '5m', '15m', '30m', '60m']:
                    temp_start = datetime.now() - timedelta(days=7 if interval == '1m' else 60)
                    st.info(f"Retrying {symbol} with shorter range: {temp_start} to {temp_end}")
        if not success:
            st.error(f"Failed to fetch data for {symbol} after {max_attempts} attempts.")
    return data

# Function to process uploaded CSV
def process_uploaded_csv(file):
    try:
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Open']):
            df = df.dropna(subset=['High', 'Low', 'Close', 'Open'])
            df[['High', 'Low', 'Close', 'Open']] = df[['High', 'Low', 'Close', 'Open']].astype(float)
            if not df.empty:
                st.info(f"Successfully loaded CSV with {len(df)} rows.")
                return {"Uploaded_Symbol": df}
            else:
                st.warning("Uploaded CSV has no valid data after cleaning.")
                return {}
        else:
            st.error(f"CSV missing required columns: {['High', 'Low', 'Close', 'Open']}")
            return {}
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return {}

# Function to classify a single candle
def classify_candle(row, min_body_rd, max_body_b):
    try:
        high_low_range = float(row['High'] - row['Low'])
        if high_low_range <= 0:
            return 'Neutral'
        
        body = abs(float(row['Close'] - row['Open']))
        body_pct = (body / high_low_range) * 100
        
        is_green = float(row['Close']) > float(row['Open'])
        is_red = float(row['Close']) < float(row['Open'])
        
        if body_pct > min_body_rd:
            if is_green:
                return 'Rally'
            elif is_red:
                return 'Drop'
        elif body_pct < max_body_b:
            return 'Base'
        
        return 'Neutral'
    except (TypeError, ValueError):
        return 'Neutral'

# Function to detect patterns with Entry/Stoploss
def detect_pattern(df, min_body_rd, max_body_b, num_bases, required_zone_type, rr_ratio, sl_buffer_pct):
    if len(df) < 2 + num_bases:
        return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
    
    df['Candle_Type'] = df.apply(lambda row: classify_candle(row, min_body_rd, max_body_b), axis=1)
    recent_candles = df.iloc[-(2 + num_bases):]
    candle_types = recent_candles['Candle_Type'].values
    
    if len(candle_types) < 2 + num_bases:
        return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
    
    leg_in = candle_types[0]
    bases = candle_types[1:1+num_bases]
    leg_out = candle_types[-1]
    
    if all(b == 'Base' for b in bases):
        pattern = None
        is_demand = False
        
        base_candles = recent_candles.iloc[1:1+num_bases]
        if base_candles.empty or base_candles['High'].isna().all() or base_candles['Low'].isna().all():
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        try:
            zone_low = float(min(recent_candles['Low']))
            zone_high = float(max(recent_candles['High']))
            base_max_high = float(max(base_candles['High']))
            base_min_low = float(min(base_candles['Low']))
        except (ValueError, TypeError):
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        if leg_in == 'Rally' and leg_out == 'Rally':
            pattern = 'RBR'
            is_demand = True
        elif leg_in == 'Rally' and leg_out == 'Drop':
            pattern = 'RBD'
            is_demand = False
        elif leg_in == 'Drop' and leg_out == 'Drop':
            pattern = 'DBD'
            is_demand = False
        elif leg_in == 'Drop' and leg_out == 'Rally':
            pattern = 'DBR'
            is_demand = True
        
        if not pattern:
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        if required_zone_type != "ALL" and ((required_zone_type == "DEMAND" and not is_demand) or (required_zone_type == "SUPPLY" and is_demand)):
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        try:
            current_price = float(df.iloc[-1]['Close'])
        except (ValueError, TypeError):
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        entry_price = base_max_high if is_demand else base_min_low
        sl_price = zone_low * (1 - sl_buffer_pct / 100) if is_demand else zone_high * (1 + sl_buffer_pct / 100)
        risk = abs(entry_price - sl_price)
        target_price = entry_price + (risk * rr_ratio) if is_demand else entry_price - (risk * rr_ratio)
        
        if is_demand:
            if current_price >= base_max_high:
                if current_price >= target_price:
                    status = 'TARGET'
                elif current_price <= sl_price:
                    status = 'STOPLOSS'
                else:
                    status = 'FRESH'
            else:
                status = 'FRESH'
        else:
            if current_price <= base_min_low:
                if current_price <= target_price:
                    status = 'TARGET'
                elif current_price >= sl_price:
                    status = 'STOPLOSS'
                else:
                    status = 'FRESH'
            else:
                status = 'FRESH'
        
        if zone_status != "ALL" and zone_status != status:
            return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'
        
        demand_score = 1.0 if is_demand else 0.0
        supply_score = 1.0 - demand_score
        return (
            f"{pattern} ({'Demand' if is_demand else 'Supply'})",
            demand_score,
            supply_score,
            1,
            entry_price,
            sl_price,
            target_price,
            status
        )
    
    return 'No Pattern', 0.0, 0.0, 0, 0.0, 0.0, 0.0, 'NONE'

# Function to create zip file with results
def create_results_zip(results_df):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add summary CSV
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        zip_file.writestr("screener_results.csv", csv_buffer.read())
        
        # Add individual CSV for each symbol
        for symbol in results_df['Symbol'].unique():
            symbol_df = results_df[results_df['Symbol'] == symbol]
            csv_buffer = BytesIO()
            symbol_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            zip_file.writestr(f"{symbol}_details.csv", csv_buffer.read())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Scan Button
if st.sidebar.button("SCAN"):
    if not symbols and not uploaded_file:
        st.warning("Please select script type, enter symbols, or upload a CSV.")
    else:
        with st.spinner(f"Processing data for {len(symbols)} symbols..."):
            data = {}
            if uploaded_file:
                data = process_uploaded_csv(uploaded_file)
            else:
                data = fetch_data(symbols, start_date, end_date, yf_interval)
        
        results = []
        for symbol, df in data.items():
            if not df.empty:
                st.info(f"Processing {symbol} with {len(df)} candles.")
                pattern, demand_score, supply_score, match, entry_price, sl_price, target_price, status = detect_pattern(
                    df, min_body_rally_drop, max_body_base, num_base_candles, zone_type, rr_ratio, sl_buffer_pct
                )
                if match > 0:
                    results.append({
                        'Symbol': symbol,
                        'Pattern': pattern,
                        'Zone_Type': 'Demand' if demand_score > 0 else 'Supply',
                        'Zone_Status': status,
                        'Current_Price': df.iloc[-1]['Close'],
                        'Entry_Price': entry_price,
                        'Stoploss_Price': sl_price,
                        'Target_Price': target_price,
                        'Demand_Score': demand_score,
                        'Supply_Score': supply_score
                    })
                else:
                    st.info(f"No matching pattern found for {symbol}.")
            else:
                st.warning(f"No data available for {symbol}.")
        
        if results:
            df_results = pd.DataFrame(results)
            st.subheader("Scan Results")
            st.dataframe(
                df_results[['Symbol', 'Pattern', 'Zone_Type', 'Zone_Status', 'Current_Price', 'Entry_Price', 'Stoploss_Price', 'Target_Price']],
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", len(df_results))
            with col2:
                demand_count = len(df_results[df_results['Demand_Score'] > 0])
                st.metric("Demand Zones", demand_count)
            with col3:
                supply_count = len(df_results[df_results['Supply_Score'] > 0])
                st.metric("Supply Zones", supply_count)
            
            # Download buttons
            zip_data = create_results_zip(df_results)
            st.download_button(
                label="Download Results as ZIP",
                data=zip_data,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            csv_buffer = BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="Download Results as CSV",
                data=csv_buffer,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No matching zones found. Try: DAILY/WEEKLY interval, 60-day range, fewer symbols, or upload a CSV with OHLC data.")

# Instructions
with st.expander("Deployment Instructions"):
    st.markdown("""
1. Create a GitHub repo and add `app.py`.
2. Create `requirements.txt` with:
   ```
   streamlit
   pandas
   yfinance>=0.2.41
   pandas_datareader>=0.10.0
   numpy
   ```
3. Deploy on Streamlit Cloud via GitHub (streamlit.io/cloud).
4. Test locally with `streamlit run app.py` to isolate issues.
5. Use CSV upload if API fails (format: Date, Open, High, Low, Close).
6. Check logs in 'Manage app' for errors.
    """)
