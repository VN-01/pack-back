import logging
import ollama
from copy import deepcopy

logging.basicConfig(level=logging.INFO)

class OllamaChat:
    def __init__(self, id: str, host: str = "http://ollama:11434"):
        logging.info(f"Initialized OllamaChat with id: {id}")
        self.id = id
        self.client = ollama.Client(host=host)
        self.instructions = None
        self.system_message = None

    def __deepcopy__(self, memo):
        logging.info(f"Performing deepcopy on OllamaChat with id: {self.id}")
        copied = OllamaChat(self.id)
        copied.instructions = deepcopy(self.instructions, memo)
        copied.system_message = deepcopy(self.system_message, memo)
        logging.info(f"Deepcopy completed for OllamaChat with id: {self.id}")
        return copied

    def get_function_schemas(self):
        logging.info(f"Returning YFinanceTools function schemas")
        return []

    def set_instructions(self, instructions: str):
        self.instructions = instructions

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def invoke(self, messages):
        logging.info(f"Returning instructions: {self.instructions}")
        logging.info(f"Returning system message: {self.system_message}")
        try:
            response = self.client.chat(model=self.id, messages=messages)
            if response is None:
                logging.error(f"Ollama chat returned None for model {self.id}")
                return None
            logging.info(f"Ollama chat response: {response}")
            return response
        except Exception as e:
            logging.error(f"Ollama chat error for model {self.id}: {str(e)}")
            return None
