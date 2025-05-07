import logging
import copy
from typing import Any, Dict, List
from agno.models.ollama import Ollama
from agno.models.message import Message
from custom_tools.mock_yfinance import MockYFinanceTools as YFinanceTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaChat:
    def __init__(self, id: str = "tinyllama", **kwargs):
        self.id = id  # Store model identifier
        self.model = Ollama(id=id, host="http://ollama:11434", **kwargs)
        self.tools = []
        self.functions = {}
        self.instructions = kwargs.get("instructions", "")
        self.system_message = kwargs.get("system_message", "You are a helpful assistant for analyzing stock data.")
        self.response = None  # Initialize response attribute
        logger.info(f"Initialized OllamaChat with id: {id}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        try:
            logger.info(f"Processing chat with messages: {messages}")
            # Convert dict messages to agno Message objects
            agno_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
            # Call Ollama's invoke method without tools
            response = self.model.invoke(agno_messages)
            logger.info(f"Ollama.invoke response: {response}")
            # Ensure response is a dict
            if not isinstance(response, dict):
                logger.error(f"Invalid response from Ollama: {response}")
                raise ValueError(f"Invalid response from Ollama: {response}")
            # Format response to match OpenAIChat's expected output
            message = response.get("message", {})
            if not isinstance(message, dict):
                logger.warning(f"Message is not a dict, converting: {message}")
                message = {"role": "assistant", "content": str(message)}
            formatted_response = {
                "choices": [{
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", "")
                    }
                }],
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                }
            }
            # Store response
            self.response = formatted_response
            logger.info(f"Stored response: {self.response}")
            return formatted_response
        except Exception as e:
            logger.error(f"OllamaChat error: {str(e)}")
            # Set a default response to avoid None
            self.response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            logger.info(f"Stored error response: {self.response}")
            raise Exception(f"OllamaChat error: {str(e)}")

    def __deepcopy__(self, memo):
        # Handle deep copy operations
        logger.info(f"Performing deepcopy on OllamaChat with id: {self.id}")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Copy attributes
        result.__dict__.update({
            'id': copy.deepcopy(self.id, memo),
            'model': Ollama(id=copy.deepcopy(self.id, memo), host="http://ollama:11434"),
            'tools': copy.deepcopy(self.tools, memo),
            'functions': copy.deepcopy(self.functions, memo),
            'instructions': copy.deepcopy(self.instructions, memo),
            'system_message': copy.deepcopy(self.system_message, memo),
            'response': copy.deepcopy(self.response, memo)
        })
        logger.info(f"Deepcopy completed for OllamaChat with id: {self.id}")
        return result

    def __getattr__(self, name):
        # Fallback for undefined attributes/methods
        logger.warning(f"Attempted to access undefined attribute/method: {name}")
        def dummy_method(*args, **kwargs):
            logger.error(f"Called undefined method {name} with args: {args}, kwargs: {kwargs}")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"Error: Method {name} is not implemented"
                    }
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        return dummy_method

    def get_functions(self) -> Dict[str, Dict[str, Any]]:
        # Return function schemas for YFinanceTools as a dictionary
        logger.info("Returning YFinanceTools function schemas")
        return {
            "get_current_stock_price": {
                "name": "get_current_stock_price",
                "description": "Retrieve the current stock price for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., NVDA)"
                        }
                    },
                    "required": ["ticker"]
                }
            },
            "get_company_info": {
                "name": "get_company_info",
                "description": "Retrieve company information for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., NVDA)"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }

    def set_tools(self, tools: List[Any]) -> None:
        # Store tools for use in chat
        logger.info(f"Setting tools: {[tool.__class__.__name__ if hasattr(tool, '__class__') else str(tool) for tool in tools]}")
        self.tools = []
        for tool in tools:
            tool_name = tool.__class__.__name__ if hasattr(tool, '__class__') else str(tool)
            func_schema = self.get_functions().get(tool_name, {})
            if not func_schema:
                logger.warning(f"No function schema found for tool: {tool_name}")
                continue
            self.tools.append({
                "type": "function",
                "function": func_schema
            })
        logger.info(f"Tools set: {self.tools}")

    def set_functions(self, functions: List[Dict[str, Any]]) -> None:
        # Store function schemas for use in chat
        logger.info(f"Setting functions: {[func.get('name', 'unknown') for func in functions]}")
        self.functions = {}
        self.tools = []
        for func in functions:
            func_name = func.get('name')
            if not func_name:
                logger.warning(f"Invalid function schema, missing name: {func}")
                continue
            self.functions[func_name] = func
            self.tools.append({"type": "function", "function": func})
            logger.info(f"Added function: {func_name}")
        logger.info(f"Functions set: {self.functions}")
        logger.info(f"Tools set: {self.tools}")

    def get_instructions_for_model(self) -> str:
        # Return instructions for the model
        logger.info(f"Returning instructions: {self.instructions}")
        return self.instructions

    def get_system_message_for_model(self) -> str:
        # Return system message for the model
        logger.info(f"Returning system message: {self.system_message}")
        return self.system_message
