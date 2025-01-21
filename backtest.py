import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, StringVar, Toplevel, Menu, Text, Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.messagebox as tkmb
from idlelib.tooltip import Hovertip
from datetime import datetime
import traceback
from tkinter import ttk

def center_screen(root, width, height):
    """Centers the tkinter window on the screen."""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width - width) / 2)
    y = int((screen_height - height) / 2)
    return width, height, x, y

def geometry_string(geometry_tuple):
    """Generates a geometry string for tkinter window positioning."""
    width, height, x, y = geometry_tuple
    return f"{width}x{height}+{x}+{y}"

def fetch_data(ticker, start, end):
    """Fetches stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data is None or data.empty:
            tkmb.showerror("Error", f"No data found for {ticker} from {start} to {end}")
            return None
        return data
    except Exception as e:
        tkmb.showerror("Error", f"Failed to fetch data: {e}")
        return None

def calculate_sma(data, window):
    """Calculates the Simple Moving Average."""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    """Calculates the Relative Strength Index."""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma_crossover_strategy(data, short_win, long_win):
    """Calculates trading signals based on SMA crossover."""
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    signals['SMA_Short'] = calculate_sma(data, short_win)
    signals['SMA_Long'] = calculate_sma(data, long_win)
    signals.loc[signals['SMA_Short'] > signals['SMA_Long'], 'Signal'] = 1.0
    signals.loc[signals['SMA_Short'] <= signals['SMA_Long'], 'Signal'] = 0.0
    signals['Position'] = signals['Signal'].diff()
    return signals

def calculate_rsi_strategy(data, oversold, overbought, rsi_win=14):
    """Calculates trading signals based on RSI levels."""
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    signals['RSI'] = calculate_rsi(data, rsi_win)
    signals.loc[(signals['RSI'].shift(1) >= oversold) & (signals['RSI'] < oversold), 'Signal'] = 1.0
    signals.loc[(signals['RSI'].shift(1) <= overbought) & (signals['RSI'] > overbought), 'Signal'] = -1.0
    signals['Position'] = signals['Signal']
    return signals

def backtest(data, signals, initial_capital=1000):
    """Performs the backtest of the trading strategy."""
    # Create a new DataFrame with copy to avoid warnings
    portfolio = pd.DataFrame(index=data.index).copy()
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = initial_capital
    portfolio['Total'] = initial_capital
    trades = []
    in_position = False
    entry_price = 0
    shares = 0

    # Determine which price column to use
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    if price_column not in data.columns:
        raise ValueError("Neither 'Adj Close' nor 'Close' column found in data")

    # Align the signals with the data
    aligned_signals = signals.reindex(data.index)
    
    for i in range(len(data)):
        current_date = data.index[i]
        current_price = float(data[price_column].iloc[i])  # Convert to float

        if i > 0:  # Copy previous values
            portfolio.iloc[i] = portfolio.iloc[i-1]  # Copy all values from previous row

        # Skip if we don't have a signal for this date
        if current_date not in aligned_signals.index or pd.isna(aligned_signals['Position'].loc[current_date]):
            continue

        current_signal = aligned_signals['Position'].loc[current_date]

        if current_signal == 1.0 and not in_position:
            entry_price = current_price
            # Calculate maximum shares we can buy with current cash
            shares = int(portfolio.iloc[i]['Cash'] // entry_price)
            if shares > 0:
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = shares
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] -= shares * entry_price
                in_position = True
                trades.append({
                    'Date': current_date,
                    'Action': 'BUY',
                    'Price': entry_price,
                    'Quantity': shares,
                    'Profit': None
                })

        elif current_signal == -1.0 and in_position:
            exit_price = current_price
            if shares > 0:
                profit = (exit_price - entry_price) * shares
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] += shares * exit_price
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = 0
                trades.append({
                    'Date': current_date,
                    'Action': 'SELL',
                    'Price': exit_price,
                    'Quantity': shares,
                    'Profit': profit
                })
                shares = 0
                in_position = False

        # Update total value
        portfolio.iloc[i, portfolio.columns.get_loc('Total')] = (
            portfolio.iloc[i]['Cash'] + portfolio.iloc[i]['Holdings'] * current_price
        )

    return portfolio, pd.DataFrame(trades)

def plot_results(data, signals, portfolio, trades, strategy_name, ticker):
    """Plots the backtesting results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{strategy_name} Backtest on {ticker}', fontsize=14, fontname='Arial')

    # Determine which price column to use
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

    # Plot price and indicators
    axes[0].plot(data[price_column], label=f'{ticker} {price_column}', color='blue')
    if 'SMA_Short' in signals.columns:
        axes[0].plot(signals['SMA_Short'], label='Short SMA', color='orange')
        axes[0].plot(signals['SMA_Long'], label='Long SMA', color='purple')
    if 'RSI' in signals.columns:
        axes[0].axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        axes[0].axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        axes[0].plot(signals['RSI'], label='RSI', color='purple', alpha=0.7)
    
    # Plot buy/sell markers using the determined price column
    axes[0].plot(signals.loc[signals['Position'] == 1.0].index, 
                 data[price_column][signals['Position'] == 1.0], 
                 '^', markersize=8, color='g', label='Buy')
    axes[0].plot(signals.loc[signals['Position'] == -1.0].index, 
                 data[price_column][signals['Position'] == -1.0], 
                 'v', markersize=8, color='r', label='Sell')
    
    axes[0].set_ylabel('Price', fontsize=12, fontname='Arial')
    axes[0].legend(prop={'family': 'Arial', 'size': 10})
    axes[0].grid(True)

    axes[1].set_ylabel('Signal/RSI', fontsize=12, fontname='Arial')
    axes[1].grid(True)
    if 'Signal' in signals.columns and 'RSI' not in signals.columns:
        axes[1].plot(signals['Signal'], label='Trading Signal', color='orange', linestyle='--')
        axes[1].legend(prop={'family': 'Arial', 'size': 10})
    elif 'RSI' in signals.columns:
        axes[1].plot(signals['RSI'], label='RSI', color='purple')
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
        axes[1].legend(prop={'family': 'Arial', 'size': 10})

    axes[2].plot(portfolio['Total'], label='Portfolio Value', color='green')
    axes[2].set_ylabel('Portfolio Value', fontsize=12, fontname='Arial')
    axes[2].set_xlabel('Date', fontsize=12, fontname='Arial')
    axes[2].legend(prop={'family': 'Arial', 'size': 10})
    axes[2].grid(True)

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(10)

    return fig

class BacktestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantitative Trading Backtester")
        self.root.resizable(False, False)

        self.style = ttk.Style(self.root)
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('TEntry', font=('Arial', 12))
        self.style.configure('TCombobox', font=('Arial', 12))
        self.style.configure('TLabelframe.Label', font=('Arial', 12, 'bold'))

        self._create_menu()
        self._create_input_frame()
        self._create_buttons()
        self._create_strategy_parameters_frame()
        self._create_status_bar()

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        self.root.geometry(geometry_string(center_screen(self.root, width, height)))
        self._update_clock()
        self.root.focus_force()

    def _create_menu(self):
        menu_bar = Menu(self.root)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Strategy Information", command=self._show_strategy_info)
        help_menu.add_command(label="Parameter Information", command=self._show_parameter_info)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        about_menu = Menu(menu_bar, tearoff=0)
        about_menu.add_command(label="Credits", command=self._show_credits)
        menu_bar.add_cascade(label="About", menu=about_menu)

        self.root.config(menu=menu_bar)

    def _create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Stock Data")
        input_frame.pack(padx=10, pady=5, fill="x", expand=True)

        # Ticker
        ttk.Label(input_frame, text="Ticker:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.ticker_var = StringVar(self.root, value="NVDA")
        self.ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var)
        self.ticker_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.ticker_entry, "Enter the stock ticker symbol (e.g., NVDA)")

        # Start Date
        ttk.Label(input_frame, text="Start Date:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.start_date_var = StringVar(self.root, value="2024-01-01")
        self.start_date_entry = ttk.Entry(input_frame, textvariable=self.start_date_var)
        self.start_date_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.start_date_entry, "Enter the start date for backtesting (YYYY-MM-DD)")

        # End Date
        ttk.Label(input_frame, text="End Date:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.end_date_var = StringVar(self.root, value="2025-01-21")
        self.end_date_entry = ttk.Entry(input_frame, textvariable=self.end_date_var)
        self.end_date_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.end_date_entry, "Enter the end date for backtesting (YYYY-MM-DD)")

        # Strategy
        ttk.Label(input_frame, text="Strategy:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.strategy_var = StringVar(self.root, value="RSI")
        self.strategy_combobox = ttk.Combobox(input_frame, textvariable=self.strategy_var, values=["SMA Crossover", "RSI"])
        self.strategy_combobox.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.strategy_combobox, "Select the trading strategy")
        self.strategy_var.trace_add('write', self._toggle_strategy_parameters)

        input_frame.columnconfigure(1, weight=1)

    def _create_strategy_parameters_frame(self):
        self.sma_params_frame = ttk.LabelFrame(self.root, text="SMA Parameters")
        self.sma_params_frame.pack(padx=10, pady=5, fill="x", expand=True)

        # Short Window
        ttk.Label(self.sma_params_frame, text="Short Window:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.short_window_var = StringVar(self.sma_params_frame, value="20")
        self.short_window_entry = ttk.Entry(self.sma_params_frame, textvariable=self.short_window_var)
        self.short_window_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.short_window_entry, "Number of days for the short-term moving average")
        Button(self.sma_params_frame, text="?", font=('Arial', 10), command=lambda: self._show_help("SMA Short Window")).grid(row=0, column=2, sticky="w", padx=5, pady=2)

        # Long Window
        ttk.Label(self.sma_params_frame, text="Long Window:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.long_window_var = StringVar(self.sma_params_frame, value="50")
        self.long_window_entry = ttk.Entry(self.sma_params_frame, textvariable=self.long_window_var)
        self.long_window_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.long_window_entry, "Number of days for the long-term moving average")
        Button(self.sma_params_frame, text="?", font=('Arial', 10), command=lambda: self._show_help("SMA Long Window")).grid(row=1, column=2, sticky="w", padx=5, pady=2)

        self.sma_params_frame.columnconfigure(1, weight=1)

        self.rsi_params_frame = ttk.LabelFrame(self.root, text="RSI Parameters")
        self.rsi_params_frame.pack(padx=10, pady=5, fill="x", expand=True)

        # Oversold Level
        ttk.Label(self.rsi_params_frame, text="Oversold Level:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.oversold_var = StringVar(self.rsi_params_frame, value="30")
        self.oversold_entry = ttk.Entry(self.rsi_params_frame, textvariable=self.oversold_var)
        self.oversold_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.oversold_entry, "Oversold level for RSI")
        Button(self.rsi_params_frame, text="?", font=('Arial', 10), command=lambda: self._show_help("RSI Oversold")).grid(row=0, column=2, sticky="w", padx=5, pady=2)

        # Overbought Level
        ttk.Label(self.rsi_params_frame, text="Overbought Level:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.overbought_var = StringVar(self.rsi_params_frame, value="70")
        self.overbought_entry = ttk.Entry(self.rsi_params_frame, textvariable=self.overbought_var)
        self.overbought_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.overbought_entry, "Overbought level for RSI")
        Button(self.rsi_params_frame, text="?", font=('Arial', 10), command=lambda: self._show_help("RSI Overbought")).grid(row=1, column=2, sticky="w", padx=5, pady=2)

        # RSI Window
        ttk.Label(self.rsi_params_frame, text="RSI Window:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        self.rsi_window_var = StringVar(self.rsi_params_frame, value="14")
        self.rsi_window_entry = ttk.Entry(self.rsi_params_frame, textvariable=self.rsi_window_var)
        self.rsi_window_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        Hovertip(self.rsi_window_entry, "Number of periods for RSI calculation")
        Button(self.rsi_params_frame, text="?", font=('Arial', 10), command=lambda: self._show_help("RSI Window")).grid(row=2, column=2, sticky="w", padx=5, pady=2)

        self.rsi_params_frame.columnconfigure(1, weight=1)

        self._toggle_strategy_parameters()

    def _create_buttons(self):
        self.run_button = ttk.Button(self.root, text="Run Backtest", command=self._run_backtest)
        self.run_button.pack(pady=10)

    def _create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side="bottom", fill="x")

        self.time_label = ttk.Label(self.status_bar, anchor="e", font=('Arial', 10))
        self.time_label.pack(side="right", padx=10)

    def _update_clock(self):
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            self.time_label.config(text=current_time)
            self.root.after(1000, self._update_clock)
        except Exception:
            pass  # Ignore errors when window is closed

    def _toggle_strategy_parameters(self, *args):
        if self.strategy_var.get() == "SMA Crossover":
            self.sma_params_frame.pack(padx=10, pady=5, fill="x", expand=True)
            self.rsi_params_frame.pack_forget()
        elif self.strategy_var.get() == "RSI":
            self.rsi_params_frame.pack(padx=10, pady=5, fill="x", expand=True)
            self.sma_params_frame.pack_forget()

    def _run_backtest(self):
        ticker = self.ticker_var.get().upper()
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        strategy = self.strategy_var.get()

        progress_window = Toplevel(self.root)
        progress_window.title("Processing")
        progress_window.resizable(False, False)
        progress_width = 300
        progress_height = 100
        progress_window.geometry(geometry_string(center_screen(progress_window, progress_width, progress_height)))
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.focus_force()

        progress_bar = ttk.Progressbar(progress_window, mode='determinate', maximum=100)
        progress_bar.pack(pady=20, padx=20, fill="x")
        progress_label = ttk.Label(progress_window, text="Fetching data...", font=('Arial', 10))
        progress_label.pack(pady=5)

        progress_bar['value'] = 0
        self.root.update()

        try:
            progress_label.config(text="Downloading stock data...")
            stock_data = fetch_data(ticker, start_date, end_date)
            if stock_data is None:
                progress_window.destroy()
                return

            progress_bar['value'] = 30
            progress_label.config(text="Calculating signals...")
            self.root.update()

            if strategy == "SMA Crossover":
                try:
                    short_win = int(self.short_window_var.get())
                    long_win = int(self.long_window_var.get())
                    trading_signals = calculate_sma_crossover_strategy(stock_data, short_win, long_win)
                except ValueError as e:
                    progress_window.destroy()
                    self._show_error_and_traceback(f"Invalid SMA window values: {e}")
                    return
            elif strategy == "RSI":
                try:
                    oversold_level = int(self.oversold_var.get())
                    overbought_level = int(self.overbought_var.get())
                    rsi_window = int(self.rsi_window_var.get())
                    trading_signals = calculate_rsi_strategy(stock_data, oversold_level, overbought_level, rsi_window)
                except ValueError as e:
                    progress_window.destroy()
                    self._show_error_and_traceback(f"Invalid RSI level values: {e}")
                    return

            progress_bar['value'] = 60
            progress_label.config(text="Performing backtest...")
            self.root.update()

            portfolio_history, trades_history = backtest(stock_data, trading_signals)

            progress_bar['value'] = 90
            progress_label.config(text="Generating results...")
            self.root.update()

            progress_window.destroy()
            self._show_backtest_results(stock_data, trading_signals, portfolio_history, trades_history, strategy)

        except Exception as e:
            progress_window.destroy()
            self._show_error_and_traceback(f"Error during backtest: {e}", traceback.format_exc())

    def _show_error_and_traceback(self, error_message, traceback_info=None):
        error_window = Toplevel(self.root)
        error_window.title("Error")
        error_window.resizable(False, False)
        error_window.geometry(geometry_string(center_screen(error_window, 600, 400)))
        error_window.focus_force()

        error_label = ttk.Label(error_window, text=error_message, font=('Arial', 12))
        error_label.pack(pady=10, padx=10)

        if traceback_info:
            traceback_label = ttk.Label(error_window, text="Detailed Traceback:", font=('Arial', 12, 'bold'))
            traceback_label.pack(pady=(10, 0), padx=10, anchor='w')

            traceback_text = Text(error_window, wrap="word", height=10, font=('Courier New', 10))
            traceback_text.insert("1.0", traceback_info)
            traceback_text.config(state="disabled")
            traceback_scroll = Scrollbar(error_window, command=traceback_text.yview)
            traceback_text.config(yscrollcommand=traceback_scroll.set)
            traceback_text.pack(pady=5, padx=10, fill="both", expand=True)
            traceback_scroll.pack(side="right", fill="y")

            # Changed from ttk.Button to regular Button for font support
            copy_button = Button(error_window, text="Copy Traceback",
                               command=lambda: self._copy_to_clipboard(traceback_info),
                               font=('Arial', 10))
            copy_button.pack(pady=10)

        ok_button = ttk.Button(error_window, text="OK", command=error_window.destroy)
        ok_button.pack(pady=10)

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        tkmb.showinfo("Copied", "Traceback copied to clipboard!")

    def _show_backtest_results(self, stock_data, trading_signals, portfolio_history, trades_history, strategy_name):
        results_window = Toplevel(self.root)
        results_window.title("Backtest Results")
        results_window.resizable(False, False)
        results_window.geometry(geometry_string(center_screen(results_window, 900, 700)))
        results_window.focus_force()

        # Pass ticker to plot_results
        figure = plot_results(stock_data, trading_signals, portfolio_history, trades_history, strategy_name, self.ticker_var.get().upper())
        canvas = FigureCanvasTkAgg(figure, master=results_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True)
        canvas.draw()

        if not trades_history.empty:
            trades_window = Toplevel(self.root)
            trades_window.title("Trades")
            trades_window.resizable(False, False)
            trades_window.geometry(geometry_string(center_screen(trades_window, 400, 500)))
            trades_window.focus_force()

            text_area = Text(trades_window, wrap="none", font=('Arial', 10))
            scrollbar_y = Scrollbar(trades_window, command=text_area.yview)
            scrollbar_x = Scrollbar(trades_window, orient="horizontal", command=text_area.xview)
            text_area.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

            scrollbar_y.pack(side="right", fill="y")
            scrollbar_x.pack(side="bottom", fill="x")
            text_area.pack(fill="both", expand=True)

            text_area.insert("1.0", trades_history.to_string())
            text_area.config(state="disabled")
        else:
            tkmb.showinfo("Information", "No trades were executed with the current strategy and parameters.")

    def _show_strategy_info(self):
        info_window = Toplevel(self.root)
        info_window.title("Strategy Information")
        info_window.resizable(False, False)
        info_window.geometry(geometry_string(center_screen(info_window, 600, 350)))
        info_window.focus_force()

        text_widget = Text(info_window, wrap="word", font=('Arial', 11))
        text_widget.insert("1.0", "Available Strategies:\n\n")
        text_widget.insert("end", "SMA Crossover: This strategy generates buy signals when the short-term moving average crosses above the long-term moving average, indicating potential upward price momentum. Conversely, sell signals are generated when the short-term moving average crosses below the long-term moving average, suggesting potential downward price momentum.\n\n")
        text_widget.insert("end", "RSI: The Relative Strength Index (RSI) strategy is used to identify overbought and oversold conditions in the market. A buy signal is generated when the RSI falls below the oversold level, suggesting the asset may be undervalued. A sell signal is generated when the RSI rises above the overbought level, suggesting the asset may be overvalued.\n")
        text_widget.config(state="disabled")
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)

    def _show_parameter_info(self):
        info_window = Toplevel(self.root)
        info_window.title("Parameter Information")
        info_window.resizable(False, False)
        info_window.geometry(geometry_string(center_screen(info_window, 600, 450)))
        info_window.focus_force()

        text_widget = Text(info_window, wrap="word", font=('Arial', 11))
        text_widget.insert("1.0", "Parameter Descriptions:\n\n")
        text_widget.insert("end", "Ticker: The stock symbol for which to perform the backtest. Examples include AAPL for Apple Inc., MSFT for Microsoft Corp., and GOOG for Alphabet Inc.\n\n")
        text_widget.insert("end", "Start Date: The date from which to begin fetching historical stock data for the backtest. The format should be YYYY-MM-DD (e.g., 2020-01-01).\n\n")
        text_widget.insert("end", "End Date: The date until which historical stock data will be fetched for the backtest. The format should be YYYY-MM-DD (e.g., 2023-12-31).\n\n")
        text_widget.insert("end", "SMA Parameters:\n")
        text_widget.insert("end", "  Short Window: The number of periods (typically days) to use for calculating the short-term Simple Moving Average. This average reacts quickly to recent price changes.\n")
        text_widget.insert("end", "  Long Window: The number of periods (typically days) to use for calculating the long-term Simple Moving Average. This average is smoother and less sensitive to short-term price fluctuations.\n\n")
        text_widget.insert("end", "RSI Parameters:\n")
        text_widget.insert("end", "  Oversold Level: The RSI value below which the stock is considered oversold, potentially indicating a buying opportunity. Common values are 30 or below.\n")
        text_widget.insert("end", "  Overbought Level: The RSI value above which the stock is considered overbought, potentially indicating a selling opportunity. Common values are 70 or above.\n")
        text_widget.insert("end", "  RSI Window: The number of periods used to calculate the Relative Strength Index (RSI). A smaller window makes the RSI more sensitive to recent price changes.\n")
        text_widget.config(state="disabled")
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)

    def _show_credits(self):
        credits_window = Toplevel(self.root)
        credits_window.title("Credits")
        credits_window.resizable(False, False)
        credits_window.geometry(geometry_string(center_screen(credits_window, 500, 500)))
        credits_window.focus_force()

        credits_text = """
Quantitative Trading Backtester (QTB)

Developed using:
- Python 3
- yfinance: For fetching financial data
- pandas: For data manipulation and analysis
- matplotlib: For creating visualizations
- Tkinter: For the graphical user interface

Development Lead: Gemini 2.0 flash thinking experimental
Bug Fixer: Claude 3.5 Sonnet
Testing: @eric15342335
Packer: PyInstaller
Data Source: Yahoo Finance

Changelog:
- v1.0: Initial release, supports SMA Crossover and RSI
        strategies with customizable parameters
        (note: RSI window is fixed at 14)
- v1.1: Added RSI window parameter for customization
        """
        credits_label = Label(credits_window, text=credits_text, justify="left", font=('Arial', 10))
        credits_label.pack(padx=20, pady=20)

    def _show_help(self, parameter):
        help_text = ""
        if parameter == "SMA Short Window":
            help_text = "The number of days used to calculate the short-term Simple Moving Average. A smaller value makes the moving average more sensitive to recent price changes."
        elif parameter == "SMA Long Window":
            help_text = "The number of days used to calculate the long-term Simple Moving Average. A larger value makes the moving average smoother and less sensitive to short-term fluctuations."
        elif parameter == "RSI Oversold":
            help_text = "The RSI level below which an asset is considered oversold, potentially indicating a buying opportunity. Common values are 20 or 30."
        elif parameter == "RSI Overbought":
            help_text = "The RSI level above which an asset is considered overbought, potentially indicating a selling opportunity. Common values are 70 or 80."
        elif parameter == "RSI Window":
            help_text = "The number of periods used to calculate the Relative Strength Index (RSI). A smaller window makes the RSI more sensitive to recent price changes, while a larger window makes it smoother."

        help_window = Toplevel(self.root)
        help_window.title(f"Help: {parameter.split()[-1]}")
        help_window.resizable(False, False)
        help_window_width = 400
        help_window_height = 200
        help_window.geometry(geometry_string(center_screen(help_window, help_window_width, help_window_height)))
        help_window.focus_force()
        help_label = Label(help_window, text=help_text, wraplength=380, justify="left", font=('Arial', 10))
        help_label.pack(padx=10, pady=10)

if __name__ == "__main__":
    root = Tk()
    app = BacktestGUI(root)
    root.mainloop()