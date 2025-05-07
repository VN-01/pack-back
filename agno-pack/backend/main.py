from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from custom_patches.patch_agent import Agent
from custom_ollama.ollamachat import OllamaChat
from custom_tools.yfinance_tools import YFinanceTools

app = FastAPI()
agents: Dict[str, Agent] = {}

class AgentData(BaseModel):
    name: str
    model: str
    tools: List[str] = []
    instructions: str = ""
    system_message: str = ""

class AgentRunInput(BaseModel):
    inputs: Dict[str, Any]

@app.post("/agents")
async def create_agent(data: AgentData) -> Dict[str, Any]:
    # Clear agents dictionary to avoid retrieving old agents
    agents.clear()
    
    tools = []
    if "YFinanceTools" in data.tools:
        tools.append(YFinanceTools())
    
    agent_id = str(hash(data.name + data.model))
    agent = Agent(
        model=OllamaChat(id=data.model),
        tools=tools,
        instructions=data.instructions,
        system_message=data.system_message
    )
    agents[agent_id] = agent
    return {"id": agent_id, "name": data.name, "status": "created"}

@app.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, run_input: AgentRunInput) -> Dict[str, Any]:
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.run(run_input.inputs)
