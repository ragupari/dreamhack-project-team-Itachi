import sqlite3
from typing import Optional
import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool

import rag
from config import settings


def read_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


agent_system_prompt = read_file(f"{settings.RESOURCES_PATH}/prompts/system_prompt.MD")

simple_agent = Agent(model="openai:gpt-4o", system_prompt=agent_system_prompt)

agent = Agent(model="openai:gpt-4o",
              system_prompt=agent_system_prompt,
              tools=[Tool(name="query_knowledge_base", function=rag.query, takes_ctx=False,
                          description="useful for when you need to answer questions about service information or services offered, availability and their costs.")])


@agent.tool_plain
async def get_cost_estimate(issue: str, plumbing_type: str) -> str:
    """Estimate the cost of cleaning a property based on the plumbing issue and property type."""
    system_prompt = read_file(f"{settings.RESOURCES_PATH}/prompts/cost_estimator_prompt.MD").format(issue=issue, plumbing_type=plumbing_type)

    cost_agent = Agent("openai:gpt-4o", system_prompt=system_prompt)
    response = await cost_agent.run(" ", model_settings={"temperature": 0.2})
    print(response.cost())
    print(response.data)
    return response.data


class ServiceRequest(BaseModel):
    name: str = Field(description="Full name of the lead")
    phone_number: str = Field(description="Contact phone number")
    email: str = Field(description="Email address")
    description: Optional[str] = Field(description="Additional description", default="")



def ensure_table_exists():
    conn = sqlite3.connect("lead_generation.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS service_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        email TEXT NOT NULL,
        description TEXT
    )
    """)
    conn.commit()
    conn.close()

@agent.tool_plain
async def register_service_request(request: ServiceRequest):
    """Registers a new service request in SQLite."""
    try:
        ensure_table_exists()  # Ensures the table exists before inserting data
        
        conn = sqlite3.connect("lead_generation.db")
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO service_requests (name, phone_number, email, description)
        VALUES (?, ?, ?, ?)
        """, (request.name, request.phone_number, request.email, request.description))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Service request registered successfully."}
    
    except sqlite3.Error as e:
        return {"status": "error", "error": str(e)}
    

@agent.tool_plain
async def get_service_requests():
    """Fetches all service requests from SQLite."""
    try:
        conn = sqlite3.connect("lead_generation.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM service_requests")
        requests = cursor.fetchall()
        
        conn.close()
        
        return {"status": "success", "data": requests}
    
    except sqlite3.Error as e:
        return {"status": "error", "error": str(e)}