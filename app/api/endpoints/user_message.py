from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, UUID4
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools.support_tools import (
    record_feature_request,
    record_feedback,
    search_kb,
    search_info_store,
    escalate_ticket
)
import json

router = APIRouter()

class Message(BaseModel):
    id: UUID4
    ticket_id: UUID4
    created_by: UUID4 | None
    bot_id: UUID4 | None
    sender_type: str
    message: str
    is_system_message: bool = False

class UserMessageRequest(BaseModel):
    messages: List[Message]

class UserMessageResponse(BaseModel):
    response: str
    escalate: bool = False
    actions: List[Dict[str, Any]] = []

# Setup LLM and agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support AI. Use your tools to help users.
    - If user asks about features, use record_feature_request
    - If user gives feedback, use record_feedback
    - Search KB and info store for relevant info
    - IMPORTANT: If you need to escalate to a human agent or mention escalation in your response, you MUST use the escalate_ticket tool first
    - ALWAYS use escalate_ticket tool BEFORE telling user you will escalate their request
    Keep responses friendly but professional."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [record_feature_request, record_feedback, search_kb, search_info_store, escalate_ticket]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@router.post("/user-message", response_model=UserMessageResponse)
async def handle_user_message(request: UserMessageRequest):
    try:
        # Get latest message
        latest_message = request.messages[-1].message if request.messages else ""
        
        # Run agent
        result = await agent_executor.ainvoke({
            "input": latest_message,
            "chat_history": request.messages[:-1] if len(request.messages) > 1 else []
        })

        # Parse tool outputs for actions
        actions = []
        output = str(result["output"])
        
        # Get actions from intermediate steps
        for step in result.get("intermediate_steps", []):
            if isinstance(step, tuple) and len(step) == 2:
                action = step[1]  # Second item in tuple is the tool output
                if isinstance(action, dict) and "action" in action:
                    actions.append(action)

        # Check if we need to escalate
        escalate = any(a.get("action") == "escalate" for a in actions)
        
        # Get response text and check for escalation words
        escalation_words = ['escalate', 'escalated', 'human agent', 'support representative', 'real person']
        response_text = str(result["output"])
        response_lower = response_text.lower()
        
        if not escalate:  # Only check text if not already escalating
            escalate = any(word.lower() in response_lower for word in escalation_words)
            
            # If found escalation words but no escalate action, add one
            if escalate and not any(a.get("action") == "escalate" for a in actions):
                actions.append({
                    "action": "escalate",
                    "reason": "Auto-escalation from response text",
                    "metadata": {
                        "priority": "high",
                        "status": "fresh",
                        "category": "general"
                    }
                })
                
                # Add escalation notice to response if not already there
                if not any(word.lower() in response_lower for word in ['escalated', 'human agent will', 'support representative will']):
                    response_text = "I'll need to escalate this to a human agent. " + response_text
        
        return UserMessageResponse(
            response=response_text,     # Use potentially modified response text
            escalate=escalate,          # Now properly set based on both action and text
            actions=actions             # Now includes escalate action if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 