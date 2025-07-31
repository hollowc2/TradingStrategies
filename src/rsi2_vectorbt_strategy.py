#!/usr/bin/env python3
"""
RSI(2) Mean Reversion Strategy Implementation using vectorbt

This implementation follows the strategy outlined in rsi_2_mean_reversion_strategy.md:
- Universe: S&P 500 stocks
- Trend Filter: Price > 200-day moving average (MA)
- Entry Signal: RSI(2) < 10
- Exit Signal: Price > 10-day moving average
- Position Sizing: Max 10 open positions, 10% capital per position
- If >10 signals, choose the 10 stocks with the lowest RSI(2)
"""

import logging
import pandas as pd
import numpy as np
import vectorbt as vbt
from pymongo import MongoClient
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from .config import StrategyConfig, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSI2VectorbtStrategy:
    """RSI(2) Mean Reversion Strategy using vectorbt for backtesting."""
    
    def __init__(self, config: StrategyConfig = None):
        """Initialize the strategy with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.client = None
        self.close_df = None
        self.portfolio = None
        
    def connect_to_mongodb(self) -> bool:
        """Connect to MongoDB and return success status."""
        try:
            self.client = MongoClient(self.config.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            logger.info("Successfully connected to MongoDB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def load_data_from_mongodb(self) -> bool:
        """Load stock data from MongoDB and pivot to wide format."""
        try:
            if not self.client:
                if not self.connect_to_mongodb():
                    return False
            
            logger.info("Loading stock data from MongoDB...")
            collection = self.client[self.config.db_name][self.config.collection_name]
            
            # Create query filter for S&P 500 symbols if available
            query_filter = {}
            if self.config.sp500_symbols:
                query_filter = {"symbol": {"$in": self.config.sp500_symbols}}
            
            # Add date filter if specified
            if self.config.start_date or self.config.end_date:
                date_filter = {}
                if self.config.start_date:
                    date_filter["$gte"] = self.config.start_date
                if self.config.end_date:
                    date_filter["$lte"] = self.config.end_date
                query_filter["date"] = date_filter
            
            # Pull data from MongoDB
            cursor = collection.find(query_filter, {
                "_id": 0, 
                "symbol": 1, 
                "date": 1, 
                "datetime": 1,
                "close": 1,
                "open": 1,
                "high": 1,
                "low": 1,
                "volume": 1
            })
            
            data_list = list(cursor)
            if not data_list:
                logger.error("No data found in MongoDB collection")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records from MongoDB")
            
            # Use 'datetime' field if available, otherwise use 'date'
            date_field = 'datetime' if 'datetime' in df.columns else 'date'
            df['date'] = pd.to_datetime(df[date_field])
            df = df.sort_values(['symbol', 'date'])
            
            # Create wide-format close price DataFrame
            self.close_df = df.pivot(index='date', columns='symbol', values='close').sort_index()
            
            # Remove columns with too much missing data
            min_data_points = max(250, self.config.ma_trend_period + 50)  # At least 250 days or MA period + buffer
            valid_symbols = self.close_df.dropna(thresh=min_data_points, axis=1).columns
            self.close_df = self.close_df[valid_symbols]
            
            # Forward fill and backward fill to handle minor gaps
            self.close_df = self.close_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Created price matrix with {len(self.close_df)} dates and {len(self.close_df.columns)} symbols")
            logger.info(f"Date range: {self.close_df.index.min()} to {self.close_df.index.max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data from MongoDB: {str(e)}")
            return False
    
    def calculate_indicators(self) -> Dict:
        """Calculate technical indicators using vectorbt."""
        try:
            logger.info("Calculating technical indicators...")
            
            # Calculate RSI(2)
            rsi = vbt.RSI.run(self.close_df, window=self.config.rsi_period)
            
            # Calculate moving averages
            ma_10 = vbt.MA.run(self.close_df, window=self.config.ma_exit_period)
            ma_200 = vbt.MA.run(self.close_df, window=self.config.ma_trend_period)
            
            indicators = {
                'rsi': rsi,
                'ma_10': ma_10,
                'ma_200': ma_200
            }
            
            logger.info("Technical indicators calculated successfully")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {str(e)}")
            return {}
    
    def generate_signals(self, indicators: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate entry and exit signals based on strategy rules."""
        try:
            logger.info("Generating trading signals...")
            
            rsi = indicators['rsi']
            ma_10 = indicators['ma_10']
            ma_200 = indicators['ma_200']
            
            # Basic entry condition: RSI(2) < 10 AND Price > 200-day MA
            basic_entries = (rsi.rsi < self.config.rsi_entry_threshold) & (self.close_df > ma_200.ma)
            
            # Exit condition: Price > 10-day MA
            exits = self.close_df > ma_10.ma
            
            logger.info(f"Generated signals - Entry opportunities: {basic_entries.sum().sum()}")
            return basic_entries, exits
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def apply_position_limit(self, basic_entries: pd.DataFrame, rsi_values: pd.DataFrame) -> pd.DataFrame:
        """Apply max 10 positions rule by selecting stocks with lowest RSI(2)."""
        try:
            logger.info("Applying position limits...")
            
            # Create final entries DataFrame
            final_entries = basic_entries.copy().astype(bool) & False
            
            for date in basic_entries.index:
                daily_signals = basic_entries.loc[date]
                signals_count = daily_signals.sum()
                
                if signals_count > 0:
                    if signals_count <= self.config.max_positions:
                        # If we have <= max_positions signals, take all
                        final_entries.loc[date, daily_signals[daily_signals].index] = True
                    else:
                        # If we have > max_positions signals, select the lowest RSI values
                        candidate_symbols = daily_signals[daily_signals].index
                        rsi_for_candidates = rsi_values.loc[date, candidate_symbols]
                        
                        # Sort by RSI (lowest first) and take top 10
                        selected_symbols = rsi_for_candidates.nsmallest(self.config.max_positions).index
                        final_entries.loc[date, selected_symbols] = True
            
            total_signals = final_entries.sum().sum()
            logger.info(f"Final entry signals after position limits: {total_signals}")
            
            return final_entries
            
        except Exception as e:
            logger.error(f"Failed to apply position limits: {str(e)}")
            return basic_entries
    
    def create_portfolio(self, entries: pd.DataFrame, exits: pd.DataFrame) -> bool:
        """Create and run the portfolio backtest."""
        try:
            logger.info("Creating portfolio backtest...")
            
            # Create portfolio using vectorbt
            self.portfolio = vbt.Portfolio.from_signals(
                close=self.close_df,
                entries=entries,
                exits=exits,
                size=self.config.position_size,  # 10% of portfolio per position
                init_cash=self.config.initial_cash,
                fees=self.config.commission,
                freq='1D'
            )
            
            logger.info("Portfolio backtest completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create portfolio: {str(e)}")
            return False
    
    def analyze_results(self) -> Dict:
        """Analyze and return portfolio performance results."""
        try:
            if not self.portfolio:
                logger.error("No portfolio available for analysis")
                return {}
            
            logger.info("Analyzing portfolio performance...")
            
            # Get basic statistics
            stats = self.portfolio.stats()
            
            # Calculate additional metrics
            returns = self.portfolio.returns()
            total_return = self.portfolio.total_return()
            sharpe_ratio = returns.vbt.returns.sharpe_ratio()
            max_drawdown = self.portfolio.drawdowns.max_drawdown()
            
            # Trade analysis
            trades = self.portfolio.trades
            win_rate = trades.win_rate if hasattr(trades, 'win_rate') else None
            
            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'final_value': self.portfolio.value().iloc[-1],
                'num_trades': len(trades.records) if hasattr(trades, 'records') else 0,
                'stats': stats
            }
            
            logger.info("Performance analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze results: {str(e)}")
            return {}
    
    def plot_results(self, save_path: str = None):
        """Plot portfolio performance and key metrics."""
        try:
            if not self.portfolio:
                logger.error("No portfolio available for plotting")
                return
            
            logger.info("Generating performance plots...")
            
            # Plot equity curve
            fig = self.portfolio.total_return().vbt.plot(
                title="RSI(2) Mean Reversion Strategy - Equity Curve"
            )
            
            if save_path:
                fig.write_html(f"{save_path}_equity_curve.html")
                logger.info(f"Equity curve saved to {save_path}_equity_curve.html")
            
            fig.show()
            
            # Plot drawdowns
            drawdowns_fig = self.portfolio.drawdowns.plot(
                title="RSI(2) Mean Reversion Strategy - Drawdowns"
            )
            
            if save_path:
                drawdowns_fig.write_html(f"{save_path}_drawdowns.html")
                logger.info(f"Drawdowns plot saved to {save_path}_drawdowns.html")
            
            drawdowns_fig.show()
            
        except Exception as e:
            logger.error(f"Failed to plot results: {str(e)}")
    
    def run_strategy(self) -> Dict:
        """Run the complete RSI(2) mean reversion strategy."""
        try:
            logger.info("Starting RSI(2) Mean Reversion Strategy...")
            
            # Step 1: Load data from MongoDB
            if not self.load_data_from_mongodb():
                logger.error("Failed to load data. Exiting.")
                return {}
            
            # Step 2: Calculate indicators
            indicators = self.calculate_indicators()
            if not indicators:
                logger.error("Failed to calculate indicators. Exiting.")
                return {}
            
            # Step 3: Generate signals
            basic_entries, exits = self.generate_signals(indicators)
            if basic_entries.empty or exits.empty:
                logger.error("Failed to generate signals. Exiting.")
                return {}
            
            # Step 4: Apply position limits
            final_entries = self.apply_position_limit(basic_entries, indicators['rsi'].rsi)
            
            # Step 5: Create portfolio
            if not self.create_portfolio(final_entries, exits):
                logger.error("Failed to create portfolio. Exiting.")
                return {}
            
            # Step 6: Analyze results
            results = self.analyze_results()
            
            # Step 7: Display results
            if results:
                print("\n" + "="*60)
                print("RSI(2) MEAN REVERSION STRATEGY RESULTS")
                print("="*60)
                print(f"Initial Capital: ${self.config.initial_cash:,.2f}")
                print(f"Final Value: ${results.get('final_value', 0):,.2f}")
                print(f"Total Return: {results.get('total_return', 0)*100:.2f}%")
                print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
                print(f"Number of Trades: {results.get('num_trades', 0)}")
                if results.get('win_rate'):
                    print(f"Win Rate: {results.get('win_rate')*100:.1f}%")
                print("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            return {}
        
        finally:
            # Close MongoDB connection
            if self.client:
                self.client.close()
                logger.info("MongoDB connection closed")


def main():
    """Main function to run the RSI(2) mean reversion strategy."""
    # You can customize the configuration here
    config = StrategyConfig()
    
    # Override specific parameters if needed
    config.rsi_entry_threshold = 10.0  # Use RSI < 10 as per strategy document
    config.mongodb_uri = 'mongodb://localhost:27017/'  # Update with your MongoDB URI
    
    # Create and run strategy
    strategy = RSI2VectorbtStrategy(config)
    results = strategy.run_strategy()
    
    # Optional: Generate plots
    if strategy.portfolio and results:
        strategy.plot_results("rsi2_strategy_results")


if __name__ == '__main__':
    main()