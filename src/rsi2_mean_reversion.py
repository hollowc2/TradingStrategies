import logging
import pandas as pd
import pandas_ta
import numpy as np
from backtesting import Backtest, Strategy
from pymongo import MongoClient
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_mongodb(uri: str = 'mongodb://192.168.0.210:27017/') -> MongoClient:
    """Connect to MongoDB and return the client."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def fetch_stock_data(client: MongoClient, db_name: str = 'stock_data', collection_name: str = 'daily_stocks') -> Dict[str, pd.DataFrame]:
    """Fetch stock data from MongoDB and return as a dictionary of DataFrames."""
    try:
        db = client[db_name]
        collection = db[collection_name]
        data = {}
        
        cursor = collection.find()
        for doc in cursor:
            symbol = doc.get('symbol')
            if not symbol:
                continue
            if symbol not in data:
                data[symbol] = []
            data[symbol].append({
                'date': doc['datetime'],
                'Open': doc['open'],
                'High': doc['high'],
                'Low': doc['low'],
                'Close': doc['close'],
                'Volume': doc['volume']
            })
        
        # Convert to DataFrames
        for symbol in data:
            df = pd.DataFrame(data[symbol])
            if df.empty:
                continue
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            data[symbol] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data from MongoDB: {str(e)}")
        return {}

class RSI2MeanReversion(Strategy):
    """RSI2 Mean Reversion Strategy for backtesting multiple stocks."""
    # Strategy parameters
    max_positions: int = 10
    position_size: float = 0.1  # 10% of capital per position
    rsi_period: int = 2
    rsi_entry: float = 10
    ma_trend_period: int = 200
    ma_exit_period: int = 10

    def init(self) -> None:
        """Initialize indicators for all stocks."""
        self.rsi = {}
        self.ma200 = {}
        self.ma10 = {}
        
        for symbol, df in self.data.items():
            if not isinstance(df, pd.DataFrame) or len(df) <= self.ma_trend_period:
                continue
            # Calculate indicators using pandas_ta
            self.rsi[symbol] = self.I(
                lambda x: pandas_ta.rsi(x, length=self.rsi_period),
                df.Close,
                name=f'RSI_{symbol}'
            )
            self.ma200[symbol] = self.I(
                lambda x: pandas_ta.sma(x, length=self.ma_trend_period),
                df.Close,
                name=f'SMA200_{symbol}'
            )
            self.ma10[symbol] = self.I(
                lambda x: pandas_ta.sma(x, length=self.ma_exit_period),
                df.Close,
                name=f'SMA10_{symbol}'
            )

    def next(self) -> None:
        """Execute trading logic for the current step."""
        # Get eligible stocks for entry
        eligible_stocks: List[Tuple[str, float]] = []
        for symbol in self.rsi:
            if len(self.data[symbol]) <= self.ma_trend_period:
                continue
            current_price = self.data[symbol].Close[-1]
            current_rsi = self.rsi[symbol][-1]
            current_ma200 = self.ma200[symbol][-1]
            
            if not any(np.isnan(val) for val in [current_rsi, current_ma200]):
                if current_price > current_ma200 and current_rsi < self.rsi_entry:
                    eligible_stocks.append((symbol, current_rsi))
        
        # Sort by RSI (lowest first) and select top candidates
        eligible_stocks.sort(key=lambda x: x[1])
        selected_stocks = eligible_stocks[:self.max_positions]
        
        # Close positions where price > 10-day MA
        for position in self.positions:
            symbol = position.data._name
            if symbol in self.ma10 and not np.isnan(self.ma10[symbol][-1]):
                if self.data[symbol].Close[-1] > self.ma10[symbol][-1]:
                    position.close()
        
        # Open new positions
        current_positions = len(self.positions)
        available_slots = self.max_positions - current_positions
        
        for symbol, _ in selected_stocks[:available_slots]:
            if symbol not in [p.data._name for p in self.positions]:
                size = self.position_size * self.equity / self.data[symbol].Close[-1]
                self.buy(data=symbol, size=size)

def run_backtest(symbol_df: Tuple[str, pd.DataFrame]) -> Optional[Dict[str, float]]:
    """Run backtest for a single stock."""
    symbol, df = symbol_df
    try:
        bt = Backtest(
            df,
            RSI2MeanReversion,
            cash=100_000,
            commission=0.002,
            exclusive_orders=True
        )
        stats = bt.run()
        return {
            'symbol': symbol,
            'sharpe': stats['Sharpe Ratio'],
            'return': stats['Return [%]'],
            'drawdown': stats['Max. Drawdown [%]']
        }
    except Exception as e:
        logger.error(f"Error backtesting {symbol}: {str(e)}")
        return None

def main() -> None:
    """Main function to execute backtests."""
    logger.info("Starting RSI2 Mean Reversion backtest...")
    
    # Connect to MongoDB
    with connect_to_mongodb() as client:
        # Fetch stock data
        logger.info("Loading stock data from MongoDB...")
        stock_data = fetch_stock_data(client)
        
        if not stock_data:
            logger.error("No data loaded. Exiting.")
            return
        
        logger.info(f"Loaded data for {len(stock_data)} stocks")
        
        # Prepare data for multiprocessing
        symbol_data_pairs = list(stock_data.items())
        
        # Run backtests using multiprocessing
        num_processes = cpu_count()
        logger.info(f"Starting backtests using {num_processes} processes...")
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_backtest, symbol_data_pairs)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Aggregate and display results
        if results:
            results_df = pd.DataFrame(results)
            avg_stats = {
                'Sharpe Ratio': results_df['sharpe'].mean(),
                'Annual Return': results_df['return'].mean(),
                'Max Drawdown': results_df['drawdown'].min()
            }
            
            print("Backtest Results:")
            print(f"Sharpe Ratio: {avg_stats['Sharpe Ratio']:.2f}")
            print(f"Annual Return: {avg_stats['Annual Return']:.2f}%")
            print(f"Max Drawdown: {avg_stats['Max Drawdown']:.2f}%")
        else:
            print("No valid backtest results obtained.")

if __name__ == '__main__':
    main()
