import logging
from agno.agent import Agent
from custom_tools.mock_yfinance import MockYFinanceTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch Agent.__init__ to log initialization
original_init = Agent.__init__

def patched_init(self, *args, **kwargs):
    logger.info(f"Agent.__init__ called with args: {args}, kwargs: {kwargs}")
    original_init(self, *args, **kwargs)
    logger.info(f"Agent initialized with model: {self.model.__class__.__name__}, tools: {[tool.__class__.__name__ if hasattr(tool, '__class__') else str(tool) for tool in (self.tools or [])]}")

Agent.__init__ = patched_init

# Patch Agent.run to log execution and Ollama.invoke
original_run = Agent.run

def patched_run(self, *args, **kwargs):
    logger.info(f"Agent.run called with args: {args}, kwargs: {kwargs}")
    try:
        # Log tool functions
        if hasattr(self.model, 'get_functions'):
            functions = self.model.get_functions()
            logger.info(f"Available functions: {list(functions.keys())}")
        if hasattr(self, 'tools'):
            logger.info(f"Tools: {[tool.__class__.__name__ if hasattr(tool, '__class__') else str(tool) for tool in (self.tools or [])]}")
        # Log before invoking model
        logger.info(f"Invoking model: {self.model.__class__.__name__} with id: {getattr(self.model, 'id', 'unknown')}")
        result = original_run(self, *args, **kwargs)
        logger.info(f"Agent.run result: {result}")
        return result
    except Exception as e:
        logger.error(f"Agent.run error: {str(e)}")
        raise

Agent.run = patched_run
