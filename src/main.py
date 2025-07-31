#!/usr/bin/env python3
"""
Main entry point for trading strategies.
Load configs and run different strategies with charts saved to charts folder.
"""

import logging
from config import StrategyConfig, DEFAULT_CONFIG
from strategies.rsi2_vectorbt_strategy import RSI3CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run trading strategies."""
    try:
        # Initialize configuration
        config = StrategyConfig()
        
        # Override any specific parameters if needed
        config.rsi_entry_threshold = 10.0
        config.mongodb_uri = 'mongodb://localhost:27017/'
        
        logger.info("Starting RSI(3) Combined Strategy...")
        
        # Create and run strategy
        strategy = RSI3CombinedStrategy(config)
        results = strategy.run_strategy()
        
        # Generate plots and save to charts folder
        if strategy.portfolio and results:
            strategy.plot_results("src/charts/rsi3_combined_strategy_final")
            logger.info("Strategy execution completed. Charts saved to src/charts/")
        
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")
        raise

if __name__ == "__main__":
    main()