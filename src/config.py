#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StrategyConfig:
    """Configuration class for RSI(2) Mean Reversion Strategy"""
    
    # MongoDB Configuration
    mongodb_uri: str = 'mongodb://localhost:27017'
    db_name: str = 'stock_data'
    collection_name: str = 'daily_stocks'
    
    # Strategy Parameters
    max_positions: int = 10
    position_size: float = 0.1  # 10% of capital per position
    rsi_period: int = 3
    rsi_entry_threshold: float = 10.0
    ma_trend_period: int = 200  # Trend filter
    ma_exit_period: int = 10    # Exit signal
    
    # Stop Loss Parameters
    use_stop_loss: bool = True
    stop_loss_type: str = 'volatility'  # 'fixed' or 'volatility'
    stop_loss_pct: float = 0.05  # 5% stop loss from entry price (if fixed)
    atr_period: int = 14  # ATR period for volatility calculation
    atr_multiplier: float = 2.5  # ATR multiplier for stop loss (2.5x ATR below entry)
    
    # Portfolio Configuration
    initial_cash: float = 100000.0
    commission: float = 0.0  # Can be added later
    
    # Data Configuration
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = None    # Format: 'YYYY-MM-DD'
    
    # S&P 500 symbols (sample list - should be updated with complete list)
    sp500_symbols: List[str] = None
    
    def __post_init__(self):
        if self.sp500_symbols is None:
            self.sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH',
                'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO',
                'COST', 'DIS', 'KO', 'TMO', 'PEP', 'WMT', 'ABT', 'NFLX', 'ADBE', 'CRM',
                'ACN', 'VZ', 'DHR', 'CMCSA', 'NKE', 'TXN', 'PM', 'NEE', 'RTX', 'QCOM',
                'HON', 'T', 'LOW', 'IBM', 'SPGI', 'UPS', 'AMD', 'LIN', 'INTC', 'CAT'
            ]

# Default configuration instance
DEFAULT_CONFIG = StrategyConfig()