import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class Agent:
    def __init__(self, model, tools=None, instructions: str = "", system_message: str = "", **kwargs):
        self.model = model
        self.tools = tools or []
        self.instructions = instructions
        self.system_message = system_message
        self.kwargs = kwargs

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logging.info(f"Invoking model: {self.model.__class__.__name__} with id: {self.model.id}")
            # Ensure the model is set to tinyllama
            if self.model.id != "tinyllama":
                logging.warning(f"Expected model 'tinyllama', but got '{self.model.id}'. Switching to tinyllama.")
                self.model.id = "tinyllama"
            
            # Invoke the model
            messages = [{"role": "system", "content": self.system_message}]
            messages.extend([{"role": "user", "content": inputs.get("message", "")}])
            result = self.model.invoke(messages)
            
            # Check if result is None
            if result is None:
                logging.error("Model invocation returned None. Check Ollama service logs for errors.")
                return {"error": "Model invocation failed, returned None."}
            
            # Process the result
            output = result.get("message", {}).get("content", "")
            return {"result": output}
        except Exception as e:
            logging.error(f"Agent.run error: {str(e)}")
            return {"error": str(e)}
