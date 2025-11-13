import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mplfinance as mpf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature engineering function from uploaded code
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature engineering for stock price prediction
    """
    x = df.copy()

    # basic returns
    x["ret_1"] = x["Close"].pct_change(1)
    x["ret_5"] = x["Close"].pct_change(5)
    x["ret_10"] = x["Close"].pct_change(10)

    # moving averages and std
    for w in [5, 10, 20, 50, 100, 200]:
        x[f"ma_{w}"] = x["Close"].rolling(w).mean()
        x[f"std_{w}"] = x["Close"].rolling(w).std()

    # EMA
    for w in [12, 26]:
        x[f"ema_{w}"] = x["Close"].ewm(span=w, adjust=False).mean()

    # RSI(14)
    change = x["Close"].diff()
    up = change.clip(lower=0)
    down = -1 * change.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    x["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    macd_fast = x["Close"].ewm(span=12, adjust=False).mean()
    macd_slow = x["Close"].ewm(span=26, adjust=False).mean()
    x["macd"] = macd_fast - macd_slow
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    ma20 = x["Close"].rolling(20).mean().astype(float)
    sd20 = x["Close"].rolling(20).std().astype(float)
    x["bb_up"] = (ma20 + 2*sd20).astype(float)
    x["bb_down"] = (ma20 - 2*sd20).astype(float)
    x["bb_w"] = ((x["bb_up"] - x["bb_down"]) / (ma20 + 1e-9)).astype(float)

    # position in window
    for w in [20, 60, 120, 252]:
        min_w = x["Close"].rolling(w).min()
        max_w = x["Close"].rolling(w).max()
        x[f"pos_{w}"] = (x["Close"] - min_w) / (max_w - min_w + 1e-9)

    # intraday spreads + gap
    if {"High","Low","Open"}.issubset(x.columns):
        x["hl_spread"] = (x["High"] - x["Low"]) / (x["Close"] + 1e-9)
        x["oc_gap"] = (x["Open"] - x["Close"].shift(1)) / (x["Close"].shift(1) + 1e-9)

    # ATR(14)
    if {"High","Low"}.issubset(x.columns):
        prev_close = x["Close"].shift(1)
        tr = pd.concat([
            x["High"] - x["Low"],
            (x["High"] - prev_close).abs(),
            (x["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        x["atr_14"] = tr.rolling(14).mean()

    # volume features
    if "Volume" in x.columns:
        close_diff = x["Close"].diff()
        move = pd.Series(np.sign(close_diff), index=close_diff.index).fillna(0.0)
        x["obv"] = (move * x["Volume"]).fillna(0).cumsum()
        x["vol_ma20"] = x["Volume"].rolling(20).mean()
        x["vol_chg5"] = x["Volume"].pct_change(5)

    # target: next-day close
    x["target"] = x["Close"].shift(-1)

    return x.dropna()

def validate_stock_symbol(symbol):
    """
    Validate if stock symbol exists and has data
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if 'regularMarketPrice' in info or 'currentPrice' in info:
            return True
        return False
    except:
        return False

def get_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_candlestick_chart(data, predictions=None, forecast_days=None):
    """
    Create interactive candlestick chart with optional predictions
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Stock Price"
        ), row=1, col=1
    )

    # Add predictions if available
    if predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=data.index[-len(predictions):],
                y=predictions,
                mode='lines',
                name='Predictions',
                line=dict(color='red', width=2)
            ), row=1, col=1
        )

    # Add forecast if available
    if forecast_days is not None and len(forecast_days) > 0:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=len(forecast_days))
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_days,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='green', width=2, dash='dash')
            ), row=1, col=1
        )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color='lightblue'
        ), row=2, col=1
    )

    fig.update_layout(
        title="Stock Price Analysis with Predictions",
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def plot_technical_indicators(data):
    """
    Create technical indicators visualization
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD', 'Moving Averages')
    )

    # Price with Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['bb_up'], name='BB Upper', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['bb_down'], name='BB Lower', line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['rsi_14'], name='RSI(14)', line=dict(color='purple')),
        row=2, col=1
    )
    # Add horizontal lines for RSI levels
    fig.add_trace(
        go.Scatter(x=data.index, y=[70]*len(data.index), mode='lines', 
                   line=dict(color='red', dash='dash'), name='RSI 70', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=[30]*len(data.index), mode='lines', 
                   line=dict(color='green', dash='dash'), name='RSI 30', showlegend=False),
        row=2, col=1
    )

    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['macd_signal'], name='MACD Signal', line=dict(color='orange')),
        row=3, col=1
    )

    # Moving Averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ma_20'], name='MA(20)', line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ma_50'], name='MA(50)', line=dict(color='red')),
        row=4, col=1
    )

    fig.update_layout(height=800, title="Technical Indicators Analysis")
    return fig

def train_models(features_df):
    """
    Train Random Forest and Gradient Boosting models
    """
    # Prepare features and target
    feature_columns = [col for col in features_df.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X = features_df[feature_columns]
    y = features_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    # Calculate metrics
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    
    results = {
        'models': {'rf': rf_model, 'gb': gb_model},
        'predictions': {'rf': rf_pred, 'gb': gb_pred},
        'test_data': {'X_test': X_test, 'y_test': y_test},
        'metrics': {
            'rf': {'mae': rf_mae, 'r2': rf_r2},
            'gb': {'mae': gb_mae, 'r2': gb_r2}
        },
        'feature_columns': feature_columns
    }
    
    return results

def predict_future_prices(model, last_features, days_ahead):
    """
    Predict future stock prices
    """
    predictions = []
    current_features = last_features.copy()
    
    for _ in range(days_ahead):
        pred = model.predict([current_features])[0]
        predictions.append(pred)
        # For simplicity, we'll use the prediction as the new close price
        # In reality, you'd want to update all features based on the new prediction
        current_features = current_features  # Keep features constant for simplicity
    
    return predictions

# Main Streamlit App
def main():
    st.title("📈 Stock Price Prediction ML Dashboard")
    st.markdown("---")

    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)")
    
    # Date range selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*2)  # Default 2 years
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        min_value=datetime(2010, 1, 1).date(),
        max_value=end_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = date_range[0]
        end_date = start_date + timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=365*2)
        end_date = end_date
    
    # Forecast days
    forecast_days = st.sidebar.slider("Days to Forecast", min_value=1, max_value=30, value=7)
    
    # Validate and fetch data
    if st.sidebar.button("Analyze Stock"):
        if not validate_stock_symbol(stock_symbol.upper()):
            st.error(f"❌ Invalid stock symbol: {stock_symbol.upper()}")
            return
        
        with st.spinner("Fetching stock data..."):
            raw_data = get_stock_data(stock_symbol.upper(), start_date, end_date)
            
        if raw_data is None or raw_data.empty:
            st.error("❌ No data available for the selected symbol and date range.")
            return
        
        # Store data in session state
        st.session_state.raw_data = raw_data
        st.session_state.stock_symbol = stock_symbol.upper()
        st.session_state.forecast_days = forecast_days

    # Main content
    if 'raw_data' in st.session_state:
        raw_data = st.session_state.raw_data
        symbol = st.session_state.stock_symbol
        forecast_days_count = st.session_state.forecast_days
        
        # Display basic stock info
        st.header(f"📊 Analysis for {symbol}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${raw_data['Close'][-1]:.2f}")
        with col2:
            daily_change = raw_data['Close'][-1] - raw_data['Close'][-2]
            daily_change_pct = (daily_change / raw_data['Close'][-2]) * 100
            st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:.2f}%")
        with col3:
            st.metric("Volume", f"{raw_data['Volume'][-1]:,.0f}")
        with col4:
            st.metric("Data Points", len(raw_data))
        
        # Feature engineering
        with st.spinner("Generating features..."):
            features_df = make_features(raw_data)
        
        if features_df.empty:
            st.error("❌ Not enough data for feature engineering. Please select a longer date range.")
            return
        
        st.success(f"✅ Generated {len(features_df.columns)} features from {len(features_df)} data points")
        
        # Train models
        st.header("🤖 Model Training & Evaluation")
        
        with st.spinner("Training machine learning models..."):
            model_results = train_models(features_df)
        
        # Display model metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Random Forest")
            rf_metrics = model_results['metrics']['rf']
            st.metric("Mean Absolute Error", f"${rf_metrics['mae']:.2f}")
            st.metric("R² Score", f"{rf_metrics['r2']:.4f}")
        
        with col2:
            st.subheader("🚀 Gradient Boosting")
            gb_metrics = model_results['metrics']['gb']
            st.metric("Mean Absolute Error", f"${gb_metrics['mae']:.2f}")
            st.metric("R² Score", f"{gb_metrics['r2']:.4f}")
        
        # Model comparison chart
        st.subheader("Model Performance Comparison")
        
        # Select best model based on R²
        best_model_name = 'rf' if rf_metrics['r2'] > gb_metrics['r2'] else 'gb'
        best_model = model_results['models'][best_model_name]
        best_predictions = model_results['predictions'][best_model_name]
        
        st.info(f"🏆 Best performing model: {'Random Forest' if best_model_name == 'rf' else 'Gradient Boosting'}")
        
        # Plot actual vs predicted
        test_dates = features_df.index[-len(model_results['test_data']['y_test']):]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=model_results['test_data']['y_test'],
            mode='lines',
            name='Actual Prices',
            line=dict(color='blue')
        ))
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=best_predictions,
            mode='lines',
            name='Predicted Prices',
            line=dict(color='red')
        ))
        fig_pred.update_layout(
            title="Actual vs Predicted Prices (Test Set)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Future predictions
        st.header("🔮 Future Price Forecast")
        
        # Get last feature row for forecasting
        last_features = features_df[model_results['feature_columns']].iloc[-1].values
        
        with st.spinner(f"Generating {forecast_days_count}-day forecast..."):
            future_predictions = predict_future_prices(best_model, last_features, forecast_days_count)
        
        # Display forecast
        forecast_dates = pd.date_range(start=raw_data.index[-1] + timedelta(days=1), periods=forecast_days_count)
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': future_predictions
        })
        
        st.subheader("📈 Forecast Results")
        st.dataframe(forecast_df.style.format({'Predicted_Price': '${:.2f}'}))
        
        # Candlestick chart with forecast
        st.subheader("📊 Price Chart with Forecast")
        candlestick_fig = create_candlestick_chart(raw_data.tail(100), best_predictions[-50:] if len(best_predictions) > 50 else best_predictions, future_predictions)
        st.plotly_chart(candlestick_fig, use_container_width=True)
        
        # Technical indicators
        st.header("📈 Technical Analysis")
        technical_fig = plot_technical_indicators(features_df.tail(100))
        st.plotly_chart(technical_fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': model_results['feature_columns'],
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Data export
        st.header("💾 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export raw data
            csv_raw = raw_data.to_csv()
            st.download_button(
                label="📥 Download Raw Data",
                data=csv_raw,
                file_name=f"{symbol}_raw_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export features
            csv_features = features_df.to_csv()
            st.download_button(
                label="📥 Download Features",
                data=csv_features,
                file_name=f"{symbol}_features.csv",
                mime="text/csv"
            )
        
        with col3:
            # Export forecast
            csv_forecast = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast",
                data=csv_forecast,
                file_name=f"{symbol}_forecast.csv",
                mime="text/csv"
            )
        
        # Feature details
        with st.expander("🔍 View All Generated Features"):
            st.write("**Technical Indicators Generated:**")
            feature_categories = {
                "Returns": [col for col in features_df.columns if col.startswith('ret_')],
                "Moving Averages": [col for col in features_df.columns if col.startswith('ma_')],
                "Standard Deviations": [col for col in features_df.columns if col.startswith('std_')],
                "EMAs": [col for col in features_df.columns if col.startswith('ema_')],
                "Bollinger Bands": [col for col in features_df.columns if col.startswith('bb_')],
                "Position Indicators": [col for col in features_df.columns if col.startswith('pos_')],
                "Other Indicators": ['rsi_14', 'macd', 'macd_signal', 'hl_spread', 'oc_gap', 'atr_14', 'obv', 'vol_ma20', 'vol_chg5']
            }
            
            for category, features in feature_categories.items():
                available_features = [f for f in features if f in features_df.columns]
                if available_features:
                    st.write(f"**{category}:** {', '.join(available_features)}")
    
    else:
        st.info("👆 Enter a stock symbol and date range in the sidebar, then click 'Analyze Stock' to get started!")
        
        # Show example usage
        st.subheader("📚 How to Use")
        st.markdown("""
        1. **Enter Stock Symbol**: Input a valid ticker symbol (e.g., AAPL, GOOGL, MSFT)
        2. **Select Date Range**: Choose the historical data period for analysis
        3. **Set Forecast Days**: Choose how many days ahead to predict (1-30 days)
        4. **Click Analyze**: The app will fetch data, generate features, and train ML models
        5. **View Results**: Explore predictions, charts, and technical indicators
        6. **Export Data**: Download results in CSV format
        
        **Features Include:**
        - 📊 Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - 🤖 Machine Learning models (Random Forest & Gradient Boosting)
        - 📈 Interactive charts with candlestick visualization
        - 🔮 Future price forecasting
        - 📥 Data export functionality
        """)

if __name__ == "__main__":
    main()
