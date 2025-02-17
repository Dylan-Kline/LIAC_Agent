import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

def plot_trading(data, save_path, now_date=None, width=3.5, opacity=0.8, path=None):
    # Extract data (using all but the last entry for dates, prices, actions;
    # returns are shifted by one)
    print(data)
    dates = data['date'][:-1]
    closing_prices = data['price'][:-1]
    returns = data['total_profit'][1:]
    actions = data['action'][:-1]

    # Determine y-axis limits based on the closing prices
    min_y = min(closing_prices)
    max_y = max(closing_prices)
    delta = max_y - min_y
    lowerbound = round(min_y - delta * 0.1, 2)
    upperbound = round(max_y + delta * 0.1, 2)
    if delta > 5:
        lowerbound = int(lowerbound)
        upperbound = int(upperbound)

    # Create a figure with two vertically stacked subplots.
    # Height ratios roughly mimic the original 40%/35% split.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 3]})

    # --- Plot 1: Adjusted Close Prices ---
    ax1.plot(dates, closing_prices, linewidth=width, alpha=opacity, label="Adj Close Prices")
    ax1.set_ylim(lowerbound, upperbound)
    ax1.set_ylabel("Price")
    ax1.grid(True)

    # Add markers for BUY and SELL actions
    # We subtract a little from the price for BUY markers so they appear slightly lower.
    buy_plotted = False
    sell_plotted = False
    for d, p, a in zip(dates, closing_prices, actions):
        if a == 'BUY':
            ax1.scatter(d, p - delta * 0.08, s=100, marker='D', color='green', zorder=5,
                        label='BUY' if not buy_plotted else "")
            buy_plotted = True
        elif a == 'SELL':
            ax1.scatter(d, p, s=150, marker='P', color='red', zorder=5,
                        label='SELL' if not sell_plotted else "")
            sell_plotted = True

    # Optionally, mark the "now_date" if provided
    if now_date is not None and now_date in dates:
        idx = dates.index(now_date)
        p_now = closing_prices[idx]
        ax1.scatter(now_date, p_now, s=120, marker='P', color='grey', zorder=5,
                    label=f'Now: {now_date}')

    ax1.legend(loc='best')
    # Rotate x-axis labels for readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # --- Plot 2: Cumulative Returns ---
    ax2.plot(dates, returns, linewidth=width, alpha=opacity, label="Cumulative Returns")
    ax2.set_ylabel("Cumulative Returns (%)")
    ax2.grid(True)
    # Append a percent sign to y-axis tick labels
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x}%'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(loc='best')

    plt.tight_layout()
    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close(fig)


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
