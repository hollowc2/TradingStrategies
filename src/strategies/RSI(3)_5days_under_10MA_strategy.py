#!/usr/bin/env python3
"""
RSI(3) 2-Days Under 10MA Strategy Implementation using vectorbt

Enhanced entry signal strategy:
- Universe: S&P 500 stocks  
- Entry Signal: Price > 200-day MA AND Price < 10-day MA for 2 consecutive days AND RSI(3) < 10
- Exit Signal: Price > 10-day moving average
- Position Sizing: Max 10 open positions, 10% capital per position
- If >10 signals, choose the 10 stocks with the lowest RSI(3)
- Stop Loss: 2.5x ATR volatility-based stop loss
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
from config import StrategyConfig, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSI3_2DaysUnder10MA_Strategy:
    """RSI(3) 2-Days Under 10MA Strategy using vectorbt for backtesting."""
    
    def __init__(self, config: StrategyConfig = None):
        """Initialize the strategy with configuration."""
        self.config = config or DEFAULT_CONFIG
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
            min_data_points = max(250, self.config.ma_trend_period + 50)  # At least 250 days or MA period + buffer
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
            
            # Calculate RSI(3)
            rsi = vbt.RSI.run(self.close_df, window=self.config.rsi_period)
            
            # Calculate moving averages
            ma_10 = vbt.MA.run(self.close_df, window=self.config.ma_exit_period)
            ma_200 = vbt.MA.run(self.close_df, window=self.config.ma_trend_period)
            
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
    
    def calculate_consecutive_days_below_ma(self, close_prices: pd.DataFrame, ma_values: pd.DataFrame, days: int = 5) -> pd.DataFrame:
        """Calculate where price has been below MA for specified consecutive days."""
        try:
            logger.info(f"Calculating {days} consecutive days below 10MA condition...")
            
            # Create boolean mask where price < MA
            below_ma = close_prices < ma_values
            
            # Initialize result DataFrame
            consecutive_below = pd.DataFrame(False, index=close_prices.index, columns=close_prices.columns)
            
            # For each symbol, calculate consecutive days below MA
            for symbol in close_prices.columns:
                symbol_below = below_ma[symbol]
                
                # Use rolling window to check if all of the last N days were below MA
                consecutive_mask = symbol_below.rolling(window=days, min_periods=days).sum() == days
                consecutive_below[symbol] = consecutive_mask
            
            total_signals = consecutive_below.sum().sum()
            logger.info(f"Found {total_signals} instances of {days} consecutive days below 10MA")
            
            return consecutive_below
            
        except Exception as e:
            logger.error(f"Failed to calculate consecutive days below MA: {str(e)}")
            return pd.DataFrame(False, index=close_prices.index, columns=close_prices.columns)
    
    def generate_signals(self, indicators: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate entry and exit signals based on strategy rules."""
        try:
            logger.info("Generating trading signals...")
            
            rsi = indicators['rsi']
            ma_10 = indicators['ma_10']
            ma_200 = indicators['ma_200']
            atr = indicators.get('atr')
            
            # Extract the actual values from vectorbt indicators and ensure they have the same column names
            rsi_values = rsi.rsi
            ma_10_values = ma_10.ma
            ma_200_values = ma_200.ma
            
            # Fix column names - vectorbt often uses integer indexes, we need symbol names
            rsi_values.columns = self.close_df.columns
            ma_10_values.columns = self.close_df.columns
            ma_200_values.columns = self.close_df.columns
            
            # Ensure all DataFrames have the same index by finding the common index
            # (accounting for the fact that moving averages will have NaN values at the beginning)
            all_indices = [self.close_df.index, rsi_values.index, ma_10_values.index, ma_200_values.index]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            # Debug info
            logger.info(f"DataFrame shapes - Close: {self.close_df.shape}, RSI: {rsi_values.shape}, MA10: {ma_10_values.shape}, MA200: {ma_200_values.shape}")
            logger.info(f"Common index length: {len(common_index)}")
            
            # Align all DataFrames to the same dimensions
            close_aligned = self.close_df.loc[common_index]
            rsi_aligned = rsi_values.loc[common_index]
            ma_10_aligned = ma_10_values.loc[common_index]
            ma_200_aligned = ma_200_values.loc[common_index]
            
            # Calculate 2 consecutive days below 10MA condition
            consecutive_below_10ma = self.calculate_consecutive_days_below_ma(close_aligned, ma_10_aligned, days=2)
            
            # Enhanced entry condition: RSI(3) < 10 AND Price > 200-day MA AND Price < 10-day MA for 2 consecutive days
            basic_entries = (rsi_aligned < self.config.rsi_entry_threshold) & (close_aligned > ma_200_aligned) & consecutive_below_10ma
            
            # Exit condition: Price > 10-day MA  
            ma_exits = close_aligned > ma_10_aligned
            
            # For volatility-based stop loss, we'll use the portfolio's sl_stop parameter with dynamic values
            # Store ATR for later use in portfolio creation
            if atr is not None:
                self.atr_values = atr.atr.copy()
                self.atr_values.columns = self.close_df.columns
                logger.info("ATR values stored for volatility-based stop loss")
            
            logger.info(f"Generated signals - Entry opportunities: {basic_entries.sum().sum()}")
            logger.info(f"Signal dimensions: {basic_entries.shape}")
            return basic_entries, ma_exits
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def apply_position_limit(self, basic_entries: pd.DataFrame, rsi_values: pd.DataFrame) -> pd.DataFrame:
        """Apply max 10 positions rule by selecting stocks with lowest RSI(3)."""
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
                        # Ensure rsi_values columns match the candidate_symbols
                        if hasattr(rsi_values, 'columns'):
                            rsi_for_candidates = rsi_values.loc[date, candidate_symbols]
                        else:
                            # If rsi_values doesn't have the right column structure, get from indicators
                            rsi_for_candidates = rsi_values.loc[date].reindex(candidate_symbols)
                        
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
            
            # Calculate position size in dollars (10% of initial cash)
            position_size_dollars = self.config.initial_cash * self.config.position_size
            logger.info(f"Position size per trade: ${position_size_dollars:,.2f}")
            
            # Create portfolio using vectorbt with stop loss if enabled
            portfolio_kwargs = {
                'close': self.close_df,
                'entries': entries,
                'exits': exits,
                'size': position_size_dollars,  # Dollar amount per position
                'init_cash': self.config.initial_cash,
                'fees': self.config.commission
            }
            
            # Add stop loss if enabled
            if self.config.use_stop_loss:
                if self.config.stop_loss_type == 'fixed':
                    portfolio_kwargs['sl_stop'] = self.config.stop_loss_pct
                    logger.info(f"Fixed stop loss enabled: {self.config.stop_loss_pct*100:.1f}%")
                elif self.config.stop_loss_type == 'volatility' and hasattr(self, 'atr_values'):
                    # Calculate dynamic stop loss percentages based on ATR
                    atr_aligned = self.atr_values.loc[self.close_df.index, self.close_df.columns]
                    # Convert ATR to percentage: (ATR * multiplier) / close_price
                    volatility_stops = (atr_aligned * self.config.atr_multiplier) / self.close_df
                    # Ensure stops are reasonable (between 1% and 20%)
                    volatility_stops = volatility_stops.clip(0.01, 0.20)
                    
                    portfolio_kwargs['sl_stop'] = volatility_stops
                    avg_stop = volatility_stops.mean().mean() * 100
                    logger.info(f"Volatility-based stop loss enabled: {self.config.atr_multiplier}x ATR (avg: {avg_stop:.1f}%)")
                else:
                    # Fallback to fixed if volatility calculation fails
                    portfolio_kwargs['sl_stop'] = self.config.stop_loss_pct
                    logger.warning("Volatility stop calculation failed, using fixed stop loss")
            
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
            
            # Basic metrics that don't require frequency
            final_value = self.portfolio.value().iloc[-1]
            initial_value = self.config.initial_cash
            total_return = (final_value - initial_value) / initial_value
            
            # Get portfolio values for calculations
            portfolio_values = self.portfolio.value()
            returns = portfolio_values.pct_change().dropna()
            
            # Calculate Sharpe ratio (assuming ~252 trading days per year)
            sharpe_ratio = None
            try:
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            except:
                pass
            
            # Calculate max drawdown
            max_drawdown = None
            try:
                running_max = portfolio_values.expanding().max()
                drawdown = (portfolio_values - running_max) / running_max
                if hasattr(drawdown, 'min'):
                    max_dd = drawdown.min()
                    # If it's a Series, get the minimum value
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
            
            # Get basic stats (if available)
            stats = None
            try:
                stats = self.portfolio.stats()
            except:
                logger.warning("Could not calculate detailed stats due to frequency issues")
            
            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'final_value': final_value,
                'num_trades': num_trades,
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
            portfolio_value = self.portfolio.value()
            # If it's a DataFrame, sum across columns to get total portfolio value
            if hasattr(portfolio_value, 'sum') and len(portfolio_value.shape) > 1:
                portfolio_value = portfolio_value.sum(axis=1)
            
            fig = portfolio_value.vbt.plot(
                title="RSI(3) 2-Days Under 10MA Strategy - Equity Curve"
            )
            
            if save_path:
                fig.write_html(f"{save_path}_equity_curve.html")
                logger.info(f"Equity curve saved to {save_path}_equity_curve.html")
            
            fig.show()
            
            # Plot drawdowns - skip if there are issues
            try:
                drawdowns = self.portfolio.drawdowns
                if hasattr(drawdowns, 'plot'):
                    drawdowns_fig = drawdowns.plot(
                        title="RSI(3) 2-Days Under 10MA Strategy - Drawdowns"
                    )
                    
                    if save_path:
                        drawdowns_fig.write_html(f"{save_path}_drawdowns.html")
                        logger.info(f"Drawdowns plot saved to {save_path}_drawdowns.html")
                    
                    drawdowns_fig.show()
            except Exception as e:
                logger.warning(f"Could not plot drawdowns: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to plot results: {str(e)}")
    
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
            
            # Debug: Check what columns are available
            logger.info(f"Available trade columns: {list(trades_df.columns)}")
            
            # Map the actual column names
            column_map = {
                'PnL': 'pnl',
                'Return': 'return_pct_raw',
                'Size': 'size',
                'Avg Entry Price': 'entry_price',
                'Avg Exit Price': 'exit_price',
                'Entry Fees': 'entry_fees',
                'Exit Fees': 'exit_fees',
                'Entry Timestamp': 'entry_timestamp',
                'Exit Timestamp': 'exit_timestamp',
                'Column': 'symbol'
            }
            
            # Rename columns for easier access
            trades_df = trades_df.rename(columns=column_map)
            
            # Calculate return percentage for each trade (Return column is already a ratio, convert to %)
            trades_df['return_pct'] = trades_df['return_pct_raw'] * 100
            
            # Calculate duration in business days (approximate)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_timestamp'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_timestamp'])
            trades_df['duration_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
            
            # Sort by return percentage
            best_trades = trades_df.nlargest(num_best, 'return_pct')
            worst_trades = trades_df.nsmallest(num_worst, 'return_pct')
            
            # Calculate summary statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            avg_win = trades_df[trades_df['pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['return_pct'].mean() if losing_trades > 0 else 0
            
            avg_win_duration = trades_df[trades_df['pnl'] > 0]['duration_days'].mean() if winning_trades > 0 else 0
            avg_loss_duration = trades_df[trades_df['pnl'] <= 0]['duration_days'].mean() if losing_trades > 0 else 0
            
            # Position sizing analysis
            avg_position_size = trades_df['size'].mean()
            position_size_std = trades_df['size'].std()
            
            analysis = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration,
                'avg_position_size': avg_position_size,
                'position_size_std': position_size_std,
                'best_trades': best_trades,
                'worst_trades': worst_trades,
                'all_trades': trades_df
            }
            
            logger.info("Trade analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze trades: {str(e)}")
            return {}
    
    def display_trade_analysis(self, analysis: Dict):
        """Display detailed trade analysis results."""
        if not analysis:
            print("No trade analysis available")
            return
        
        print("\n" + "="*80)
        print("DETAILED TRADE ANALYSIS")
        print("="*80)
        
        # Summary statistics
        print(f"Total Trades: {analysis['total_trades']}")
        print(f"Winning Trades: {analysis['winning_trades']} ({analysis['win_rate']*100:.1f}%)")
        print(f"Losing Trades: {analysis['losing_trades']}")
        print(f"Average Win: {analysis['avg_win_pct']:.2f}%")
        print(f"Average Loss: {analysis['avg_loss_pct']:.2f}%")
        print(f"Average Win Duration: {analysis['avg_win_duration']:.1f} days")
        print(f"Average Loss Duration: {analysis['avg_loss_duration']:.1f} days")
        print(f"Average Position Size: ${analysis['avg_position_size']:,.2f}")
        print(f"Position Size Std Dev: ${analysis['position_size_std']:,.2f}")
        
        # Risk-Reward Ratio
        if analysis['avg_loss_pct'] != 0:
            risk_reward = abs(analysis['avg_win_pct'] / analysis['avg_loss_pct'])
            print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        
        print("="*80)
    
    def run_strategy(self) -> Dict:
        """Run the complete RSI(3) 2-Days Under 10MA strategy."""
        try:
            logger.info("Starting RSI(3) 2-Days Under 10MA Strategy...")
            
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
            # Use the aligned RSI values for position limiting
            rsi_for_limits = indicators['rsi'].rsi.copy()
            rsi_for_limits.columns = self.close_df.columns
            final_entries = self.apply_position_limit(basic_entries, rsi_for_limits)
            
            # Step 5: Create portfolio
            if not self.create_portfolio(final_entries, exits):
                logger.error("Failed to create portfolio. Exiting.")
                return {}
            
            # Step 6: Analyze results
            results = self.analyze_results()
            
            # Step 6.5: Detailed trade analysis
            trade_analysis = self.analyze_trades()
            
            # Step 7: Display results
            if results:
                print("\n" + "="*60)
                print("RSI(3) 2-DAYS UNDER 10MA STRATEGY RESULTS")
                print("="*60)
                print(f"Initial Capital: ${self.config.initial_cash:,.2f}")
                
                final_value = results.get('final_value', 0)
                try:
                    # Handle pandas Series or numpy values
                    if hasattr(final_value, 'iloc'):
                        final_value = final_value.iloc[-1] if len(final_value) > 0 else 0
                    elif hasattr(final_value, 'item'):
                        final_value = final_value.item()
                    print(f"Final Value: ${float(final_value):,.2f}")
                except:
                    print(f"Final Value: {final_value}")
                
                total_return = results.get('total_return', 0)
                try:
                    if hasattr(total_return, 'iloc'):
                        total_return = total_return.iloc[-1] if len(total_return) > 0 else 0
                    elif hasattr(total_return, 'item'):
                        total_return = total_return.item()
                    print(f"Total Return: {float(total_return)*100:.2f}%")
                except:
                    print(f"Total Return: {total_return}")
                
                sharpe_ratio = results.get('sharpe_ratio')
                if sharpe_ratio is not None:
                    try:
                        if hasattr(sharpe_ratio, 'item'):
                            sharpe_ratio = sharpe_ratio.item()
                        print(f"Sharpe Ratio: {float(sharpe_ratio):.3f}")
                    except:
                        print(f"Sharpe Ratio: {sharpe_ratio}")
                else:
                    print("Sharpe Ratio: N/A")
                
                max_drawdown = results.get('max_drawdown')
                if max_drawdown is not None:
                    try:
                        if hasattr(max_drawdown, 'item'):
                            max_drawdown = max_drawdown.item()
                        print(f"Max Drawdown: {float(max_drawdown)*100:.2f}%")
                    except:
                        print(f"Max Drawdown: {max_drawdown}")
                else:
                    print("Max Drawdown: N/A")
                
                print(f"Number of Trades: {results.get('num_trades', 0)}")
                
                win_rate = results.get('win_rate')
                if win_rate is not None:
                    try:
                        if hasattr(win_rate, 'mean'):
                            # If it's a Series, get the mean win rate across all symbols
                            avg_win_rate = win_rate.mean()
                            print(f"Average Win Rate: {float(avg_win_rate)*100:.1f}%")
                        elif hasattr(win_rate, 'item'):
                            win_rate = win_rate.item()
                            print(f"Win Rate: {float(win_rate)*100:.1f}%")
                        else:
                            print(f"Win Rate: {float(win_rate)*100:.1f}%")
                    except:
                        print("Win Rate: Available in detailed results")
                
                print("="*60)
            
            # Step 8: Display detailed trade analysis
            if trade_analysis:
                self.display_trade_analysis(trade_analysis)
            
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
    """Main function to run the RSI(3) 2-Days Under 10MA strategy."""
    # You can customize the configuration here
    config = StrategyConfig()
    
    # Override specific parameters if needed
    config.rsi_entry_threshold = 10.0  # Use RSI < 10 as per strategy document
    config.mongodb_uri = 'mongodb://localhost:27017/'  # Update with your MongoDB URI
    
    # Create and run strategy
    strategy = RSI3_2DaysUnder10MA_Strategy(config)
    results = strategy.run_strategy()
    
    # Optional: Generate plots
    if strategy.portfolio and results:
        strategy.plot_results("rsi3_2days_under_10ma_strategy_results")


if __name__ == '__main__':
    main()