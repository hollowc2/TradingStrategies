### use main to load configs and run the different strategies
## charts should be saved into charts folder    
##  

from config import Config
from strategies.rsi2_vectorbt_strategy import RSI2VectorBT

def main():
    config = Config()
    rsi2_vectorbt_strategy = RSI2VectorBT(config)
    rsi2_vectorbt_strategy.run()

if __name__ == "__main__":
    main()