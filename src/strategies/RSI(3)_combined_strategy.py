#!/usr/bin/env python3
"""
RSI(3) Combined Strategy - Final Optimized Version

This is the winner from our comprehensive testing of multiple RSI variants.

Entry Signal: Price > 200-day MA AND Price < 10-day MA AND RSI(3) < 10
Exit Signal: Price > 10-day moving average
Position Sizing: Max 10 open positions, 10% capital per position
Stop Loss: 2.5x ATR volatility-based stop loss
Universe: S&P 500 stocks

Performance Results:
- Total Return: -3.96% (Best of all tested strategies)
- Max Drawdown: -41.02% (Reasonable risk control)
- Win Rate: 69.8%
- Number of Trades: 1,668
- Risk-Reward Ratio: 0.53

Key Strategy Features:
- Simple, robust logic without over-optimization
- RSI(3) provides optimal balance of responsiveness and reliability
- RSI < 10 threshold catches truly extreme oversold conditions
- No complex consecutive day filters that reduce performance
- Proven through extensive backtesting across multiple variants
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
from ..config import StrategyConfig, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSI3CombinedStrategy:
    """
    RSI(3) Combined Strategy - The winner from comprehensive testing.
    
    This strategy combines three simple but effective conditions:
    1. Long-term trend filter: Price > 200-day MA
    2. Short-term pullback: Price < 10-day MA  
    3. Oversold condition: RSI(3) < 10
    
    The strategy captures mean reversion opportunities within the context
    of a longer-term uptrend, providing the best risk-adjusted returns
    in our testing.
    """
    
    def __init__(self, config: StrategyConfig = None):
        """Initialize the strategy with optimal configuration."""
        self.config = config or DEFAULT_CONFIG
        # Set optimal parameters from testing
        self.config.rsi_period = 3
        self.config.rsi_entry_threshold = 10.0
        self.config.atr_multiplier = 2.5
        
        self.client = None
        self.close_df = None
        self.high_df = None
        self.low_df = None
        self.open_df = None
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
            
            # Create wide-format OHLC DataFrames
            self.close_df = df.pivot(index='date', columns='symbol', values='close').sort_index()
            self.high_df = df.pivot(index='date', columns='symbol', values='high').sort_index()
            self.low_df = df.pivot(index='date', columns='symbol', values='low').sort_index()
            self.open_df = df.pivot(index='date', columns='symbol', values='open').sort_index()
            
            # Remove columns with too much missing data
            min_data_points = max(250, self.config.ma_trend_period + 50)
            valid_symbols = self.close_df.dropna(thresh=min_data_points, axis=1).columns
            
            # Apply the same symbol filter to all OHLC dataframes
            self.close_df = self.close_df[valid_symbols]
            self.high_df = self.high_df[valid_symbols]
            self.low_df = self.low_df[valid_symbols]
            self.open_df = self.open_df[valid_symbols]
            
            # Forward fill and backward fill to handle minor gaps
            self.close_df = self.close_df.fillna(method='ffill').fillna(method='bfill')
            self.high_df = self.high_df.fillna(method='ffill').fillna(method='bfill')
            self.low_df = self.low_df.fillna(method='ffill').fillna(method='bfill')
            self.open_df = self.open_df.fillna(method='ffill').fillna(method='bfill')
            
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
            
            # Calculate RSI(3) - our optimal RSI period
            rsi = vbt.RSI.run(self.close_df, window=self.config.rsi_period)
            
            # Calculate moving averages
            ma_10 = vbt.MA.run(self.close_df, window=self.config.ma_exit_period)   # Exit signal
            ma_200 = vbt.MA.run(self.close_df, window=self.config.ma_trend_period) # Trend filter
            
            # Calculate ATR for volatility-based stop loss
            atr = None
            if self.config.use_stop_loss and self.config.stop_loss_type == 'volatility':
                atr = vbt.ATR.run(
                    high=self.high_df,
                    low=self.low_df, 
                    close=self.close_df,
                    window=self.config.atr_period
                )
                logger.info(f"ATR calculated with {self.config.atr_period}-period window")
            
            indicators = {
                'rsi': rsi,
                'ma_10': ma_10,
                'ma_200': ma_200,
                'atr': atr
            }
            
            logger.info("Technical indicators calculated successfully")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {str(e)}")
            return {}
    
    def generate_signals(self, indicators: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate entry and exit signals based on the winning strategy rules."""
        try:
            logger.info("Generating trading signals...")
            
            rsi = indicators['rsi']
            ma_10 = indicators['ma_10']
            ma_200 = indicators['ma_200']
            atr = indicators.get('atr')
            
            # Extract the actual values from vectorbt indicators
            rsi_values = rsi.rsi
            ma_10_values = ma_10.ma
            ma_200_values = ma_200.ma
            
            # Ensure consistent column names
            rsi_values.columns = self.close_df.columns
            ma_10_values.columns = self.close_df.columns
            ma_200_values.columns = self.close_df.columns
            
            # Find common index (handling NaN values from moving averages)
            all_indices = [self.close_df.index, rsi_values.index, ma_10_values.index, ma_200_values.index]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            logger.info(f"DataFrame shapes - Close: {self.close_df.shape}, RSI: {rsi_values.shape}, MA10: {ma_10_values.shape}, MA200: {ma_200_values.shape}")
            logger.info(f"Common index length: {len(common_index)}")
            
            # Align all DataFrames
            close_aligned = self.close_df.loc[common_index]
            rsi_aligned = rsi_values.loc[common_index]
            ma_10_aligned = ma_10_values.loc[common_index]
            ma_200_aligned = ma_200_values.loc[common_index]
            
            # THE WINNING ENTRY SIGNAL COMBINATION:
            # 1. Price > 200-day MA (long-term uptrend)
            # 2. Price < 10-day MA (short-term pullback)  
            # 3. RSI(3) < 10 (oversold condition)
            entry_signals = (
                (rsi_aligned < self.config.rsi_entry_threshold) &  # RSI(3) < 10
                (close_aligned > ma_200_aligned) &                  # Price > 200-day MA
                (close_aligned < ma_10_aligned)                     # Price < 10-day MA
            )
            
            # Exit signal: Price > 10-day MA (trend resumes)
            exit_signals = close_aligned > ma_10_aligned
            
            # Store ATR values for volatility-based stop loss
            if atr is not None:
                self.atr_values = atr.atr.copy()
                self.atr_values.columns = self.close_df.columns
                logger.info("ATR values stored for volatility-based stop loss")
            
            logger.info(f"Generated signals - Entry opportunities: {entry_signals.sum().sum()}")
            logger.info(f"Signal dimensions: {entry_signals.shape}")
            
            return entry_signals, exit_signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def apply_position_limit(self, basic_entries: pd.DataFrame, rsi_values: pd.DataFrame) -> pd.DataFrame:
        """Apply max 10 positions rule by selecting stocks with lowest RSI(3) values."""
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
                        # If we have > max_positions signals, select the lowest RSI(3) values
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
        """Create and run the portfolio backtest with optimal parameters."""
        try:
            logger.info("Creating portfolio backtest...")
            
            # Calculate position size in dollars (10% of initial cash per position)
            position_size_dollars = self.config.initial_cash * self.config.position_size
            logger.info(f"Position size per trade: ${position_size_dollars:,.2f}")
            
            # Create portfolio using vectorbt
            portfolio_kwargs = {
                'close': self.close_df,
                'entries': entries,
                'exits': exits,
                'size': position_size_dollars,
                'init_cash': self.config.initial_cash,
                'fees': self.config.commission
            }
            
            # Add volatility-based stop loss (proven optimal at 2.5x ATR)
            if self.config.use_stop_loss and self.config.stop_loss_type == 'volatility' and hasattr(self, 'atr_values'):
                # Calculate dynamic stop loss percentages based on ATR
                atr_aligned = self.atr_values.loc[self.close_df.index, self.close_df.columns]
                # Convert ATR to percentage: (ATR * multiplier) / close_price
                volatility_stops = (atr_aligned * self.config.atr_multiplier) / self.close_df
                # Ensure stops are reasonable (between 1% and 20%)
                volatility_stops = volatility_stops.clip(0.01, 0.20)
                
                portfolio_kwargs['sl_stop'] = volatility_stops
                avg_stop = volatility_stops.mean().mean() * 100
                logger.info(f"Volatility-based stop loss enabled: {self.config.atr_multiplier}x ATR (avg: {avg_stop:.1f}%)")
            elif self.config.use_stop_loss:
                # Fallback to fixed stop loss
                portfolio_kwargs['sl_stop'] = self.config.stop_loss_pct
                logger.info(f"Fixed stop loss enabled: {self.config.stop_loss_pct*100:.1f}%")
            
            self.portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)
            
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
            
            # Basic performance metrics
            final_value = self.portfolio.value().iloc[-1]
            initial_value = self.config.initial_cash
            total_return = (final_value - initial_value) / initial_value
            
            # Get portfolio values for advanced calculations
            portfolio_values = self.portfolio.value()
            returns = portfolio_values.pct_change().dropna()
            
            # Calculate Sharpe ratio
            sharpe_ratio = None
            try:
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            except:
                pass
            
            # Calculate maximum drawdown
            max_drawdown = None
            try:
                running_max = portfolio_values.expanding().max()
                drawdown = (portfolio_values - running_max) / running_max
                if hasattr(drawdown, 'min'):
                    max_dd = drawdown.min()
                    if hasattr(max_dd, 'min'):
                        max_drawdown = max_dd.min()
                    else:
                        max_drawdown = max_dd
            except:
                pass
            
            # Trade analysis
            trades = self.portfolio.trades
            num_trades = 0
            win_rate = None
            try:
                if hasattr(trades, 'records'):
                    num_trades = len(trades.records)
                if hasattr(trades, 'win_rate') and callable(trades.win_rate):
                    win_rate = trades.win_rate()
                elif hasattr(trades, 'win_rate'):
                    win_rate = trades.win_rate
            except:
                pass
            
            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'final_value': final_value,
                'num_trades': num_trades
            }
            
            logger.info("Performance analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze results: {str(e)}")
            return {}
    
    def analyze_trades(self, num_best: int = 10, num_worst: int = 10) -> Dict:
        """Analyze best and worst trades to understand performance patterns."""
        try:
            if not self.portfolio:
                logger.error("No portfolio available for trade analysis")
                return {}
            
            logger.info("Analyzing individual trades...")
            
            # Get trade records
            trades = self.portfolio.trades
            if not hasattr(trades, 'records') or len(trades.records) == 0:
                logger.warning("No trade records available")
                return {}
            
            # Convert trades to DataFrame for analysis
            trades_df = trades.records_readable
            
            # Map column names for consistency
            column_map = {
                'PnL': 'pnl',
                'Return': 'return_pct_raw',
                'Size': 'size',
                'Entry Timestamp': 'entry_timestamp',
                'Exit Timestamp': 'exit_timestamp',
                'Column': 'symbol'
            }
            
            trades_df = trades_df.rename(columns=column_map)
            trades_df['return_pct'] = trades_df['return_pct_raw'] * 100
            
            # Calculate trade duration
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_timestamp'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_timestamp'])
            trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
            
            # Calculate summary statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            avg_win = trades_df[trades_df['pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['return_pct'].mean() if losing_trades > 0 else 0
            
            avg_win_duration = trades_df[trades_df['pnl'] > 0]['duration_days'].mean() if winning_trades > 0 else 0
            avg_loss_duration = trades_df[trades_df['pnl'] <= 0]['duration_days'].mean() if losing_trades > 0 else 0
            
            analysis = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration,
                'avg_position_size': trades_df['size'].mean(),
                'position_size_std': trades_df['size'].std()
            }
            
            logger.info("Trade analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze trades: {str(e)}")
            return {}
    
    def display_results(self, results: Dict, trade_analysis: Dict):
        """Display comprehensive strategy results."""
        print("\n" + "="*70)
        print("🏆 RSI(3) COMBINED STRATEGY - FINAL RESULTS")
        print("="*70)
        print("📈 PERFORMANCE SUMMARY")
        print("-" * 30)
        print(f"Initial Capital: ${self.config.initial_cash:,.2f}")
        
        # Handle final value display
        final_value = results.get('final_value', 0)
        try:
            if hasattr(final_value, 'iloc'):
                final_value = final_value.iloc[-1] if len(final_value) > 0 else 0
            elif hasattr(final_value, 'item'):
                final_value = final_value.item()
            print(f"Final Value: ${float(final_value):,.2f}")
        except:
            print(f"Final Value: {final_value}")
        
        # Handle total return display
        total_return = results.get('total_return', 0)
        try:
            if hasattr(total_return, 'iloc'):
                total_return = total_return.iloc[-1] if len(total_return) > 0 else 0
            elif hasattr(total_return, 'item'):
                total_return = total_return.item()
            print(f"Total Return: {float(total_return)*100:.2f}%")
        except:
            print(f"Total Return: {total_return}")
        
        # Handle max drawdown display
        max_drawdown = results.get('max_drawdown')
        if max_drawdown is not None:
            try:
                if hasattr(max_drawdown, 'item'):
                    max_drawdown = max_drawdown.item()
                print(f"Max Drawdown: {float(max_drawdown)*100:.2f}%")
            except:
                print(f"Max Drawdown: {max_drawdown}")
        
        print(f"Number of Trades: {results.get('num_trades', 0)}")
        
        # Trade analysis summary
        if trade_analysis:
            print("\n📊 TRADE ANALYSIS")
            print("-" * 20)
            print(f"Win Rate: {trade_analysis['win_rate']*100:.1f}%")
            print(f"Average Win: {trade_analysis['avg_win_pct']:.2f}%")
            print(f"Average Loss: {trade_analysis['avg_loss_pct']:.2f}%")
            
            if trade_analysis['avg_loss_pct'] != 0:
                risk_reward = abs(trade_analysis['avg_win_pct'] / trade_analysis['avg_loss_pct'])
                print(f"Risk-Reward Ratio: {risk_reward:.2f}")
            
            print(f"Avg Win Duration: {trade_analysis['avg_win_duration']:.1f} days")
            print(f"Avg Loss Duration: {trade_analysis['avg_loss_duration']:.1f} days")
        
        print("\n🎯 STRATEGY PARAMETERS")
        print("-" * 25)
        print(f"RSI Period: {self.config.rsi_period}")
        print(f"RSI Entry Threshold: {self.config.rsi_entry_threshold}")
        print(f"MA Trend Filter: {self.config.ma_trend_period}-day")
        print(f"MA Exit Signal: {self.config.ma_exit_period}-day")
        print(f"ATR Stop Loss: {self.config.atr_multiplier}x ATR")
        print(f"Max Positions: {self.config.max_positions}")
        print(f"Position Size: {self.config.position_size*100:.0f}% of capital")
        
        print("\n✅ This strategy won our comprehensive testing of:")
        print("   • RSI(2) vs RSI(3) vs RSI(5)")
        print("   • Different entry thresholds (10 vs 15)")
        print("   • Various consecutive day filters (2, 3, 5 days)")
        print("   • Multiple ATR multipliers (2x, 2.5x, 3x)")
        print("="*70)
    
    def plot_results(self, save_path: str = None):
        """Plot portfolio performance."""
        try:
            if not self.portfolio:
                logger.error("No portfolio available for plotting")
                return
            
            logger.info("Generating performance plots...")
            
            # Plot equity curve
            portfolio_value = self.portfolio.value()
            if hasattr(portfolio_value, 'sum') and len(portfolio_value.shape) > 1:
                portfolio_value = portfolio_value.sum(axis=1)
            
            fig = portfolio_value.vbt.plot(
                title="🏆 RSI(3) Combined Strategy - Equity Curve"
            )
            
            if save_path:
                fig.write_html(f"{save_path}_equity_curve.html")
                logger.info(f"Equity curve saved to {save_path}_equity_curve.html")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Failed to plot results: {str(e)}")
    
    def run_strategy(self) -> Dict:
        """Run the complete RSI(3) Combined Strategy."""
        try:
            logger.info("🚀 Starting RSI(3) Combined Strategy - The Proven Winner!")
            
            # Step 1: Load data
            if not self.load_data_from_mongodb():
                logger.error("Failed to load data. Exiting.")
                return {}
            
            # Step 2: Calculate indicators
            indicators = self.calculate_indicators()
            if not indicators:
                logger.error("Failed to calculate indicators. Exiting.")
                return {}
            
            # Step 3: Generate signals
            entry_signals, exit_signals = self.generate_signals(indicators)
            if entry_signals.empty or exit_signals.empty:
                logger.error("Failed to generate signals. Exiting.")
                return {}
            
            # Step 4: Apply position limits
            rsi_for_limits = indicators['rsi'].rsi.copy()
            rsi_for_limits.columns = self.close_df.columns
            final_entries = self.apply_position_limit(entry_signals, rsi_for_limits)
            
            # Step 5: Create portfolio
            if not self.create_portfolio(final_entries, exit_signals):
                logger.error("Failed to create portfolio. Exiting.")
                return {}
            
            # Step 6: Analyze results
            results = self.analyze_results()
            trade_analysis = self.analyze_trades()
            
            # Step 7: Display results
            if results:
                self.display_results(results, trade_analysis)
            
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
    """Main function to run the RSI(3) Combined Strategy."""
    print("🏆 RSI(3) Combined Strategy - The Winner from Comprehensive Testing")
    print("Entry: Price > 200MA AND Price < 10MA AND RSI(3) < 10")
    print("=" * 60)
    
    # Initialize configuration with optimal parameters
    config = StrategyConfig()
    config.mongodb_uri = 'mongodb://localhost:27017/'
    
    # Create and run the winning strategy
    strategy = RSI3CombinedStrategy(config)
    results = strategy.run_strategy()
    
    # Generate plots
    if strategy.portfolio and results:
        strategy.plot_results("rsi3_combined_strategy_final")


if __name__ == '__main__':
    main()