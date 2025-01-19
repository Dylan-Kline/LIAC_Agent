import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import pandas_ta as ta
import numpy as np

def plot_kline(df, title, save_path, now_date, mode="train"):
    
    # Convert date to useable date
    now_date = pd.to_datetime(now_date)
    
    # Filter data for the training mode
    if mode != "train":
        df = df[df.index <= now_date]

    # Calculate indicators
    df['sma_5'] = ta.sma(df["close"], length=5)
    bbands = ta.bbands(df["close"], length=5)
    df['bbl'] = bbands.iloc[:, 0]
    df['bbu'] = bbands.iloc[:, 2]
    df['bbp'] = bbands.iloc[:, 4]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Candlestick plot
    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Plot the candlestick body
        ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5)  # Wicks
        ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=6)  # Body

    # Plot indicators
    ax.plot(df.index, df['sma_5'], label="SMA 5", color="blue", linewidth=1.5)
    ax.plot(df.index, df['bbl'], label="BBL", color="green", linewidth=1, linestyle='--')
    ax.plot(df.index, df['bbu'], label="BBU", color="yellow", linewidth=1, linestyle='--')

    # highlight current date
    if now_date in df.index:
        ax.scatter(now_date, df.loc[now_date, 'high'], color="grey", s=100, label="Today's Date", marker='o')

    # Set axis labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)

    # Format date axis
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Add legend
    ax.legend()

    # Save the figure
    print("saving")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example DataFrame
    # Create example data for 50 days
    np.random.seed(42)  # For reproducibility
    days = 50
    date_range = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Generate random prices for open, high, low, and close
    open_prices = np.random.uniform(100, 200, size=days)
    high_prices = open_prices + np.random.uniform(0, 20, size=days)
    low_prices = open_prices - np.random.uniform(0, 20, size=days)
    close_prices = low_prices + np.random.uniform(0, high_prices - low_prices)

    # Generate random volume data
    volume = np.random.randint(1000, 10000, size=days)
    
    # Create the DataFrame
    kline_data = pd.DataFrame({
        "timestamp": date_range,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })
    df = pd.DataFrame(kline_data).set_index("timestamp")

    # Save plot
    plot_kline(df, "Kline Chart", "kline_chart.png", now_date="2023-01-25")
