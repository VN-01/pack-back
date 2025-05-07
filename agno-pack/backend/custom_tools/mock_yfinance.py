import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockYFinanceTools:
    def __init__(self):
        logger.info("Initialized MockYFinanceTools")

    def get_current_stock_price(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"Mock fetching current stock price for {ticker}")
        return {
            "ticker": ticker,
            "price": 123.45,  # Mock price
            "currency": "USD"
        }

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"Mock fetching company info for {ticker}")
        return {
            "ticker": ticker,
            "name": f"{ticker} Inc.",
            "sector": "Technology"
        }
