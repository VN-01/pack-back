from fastapi import FastAPI, HTTPException
from agno.agent import Agent
from custom_ollama import OllamaChat
from agno.tools.yfinance import YFinanceTools
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import custom_patches.patch_agent

load_dotenv()  # Load environment variables

app = FastAPI()

class AgentCreate(BaseModel):
    name: str
    model: str
    tools: List[str]
    instructions: str

class AgentRun(BaseModel):
    inputs: Dict[str, Any]

@app.post("/agents")
async def create_agent(agent: AgentCreate):
    try:
        # Initialize tools based on input
        tools = []
        if "YFinanceTools" in agent.tools:
            tools.append(YFinanceTools(stock_price=True, company_info=True))
        # Initialize model
        model = OllamaChat(id=agent.model)
        agno_agent = Agent(
            model=model,
            tools=tools,
            instructions=[agent.instructions],
            markdown=True
        )
        return {"id": id(agno_agent), "name": agent.name, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agents/{agent_id}/run")
async def run_agent(agent_id: int, run: AgentRun):
    try:
        # Simulate agent lookup (in production, use a database)
        model = OllamaChat(id="phi3")
        agno_agent = Agent(
            model=model,
            tools=[YFinanceTools(stock_price=True, company_info=True)],
            instructions=["Provide concise responses in markdown"],
            markdown=True
        )
        result = agno_agent.print_response(**run.inputs, stream=False)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sessions")
async def get_sessions():
    return {"sessions": []}  # Placeholder for session monitoring
