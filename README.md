# ForexVision

Complete End-to-End Machine Learning Forex Trading System.

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Configure API keys in config/config.yaml.

3. Run Backtest:
   python main.py --mode backtest --pairs EURUSD=X GBPUSD=X

4. Run Paper Trading:
   python main.py --mode paper_trade --pairs EURUSD=X

5. Run Dashboard:
   streamlit run dashboard/app.py
