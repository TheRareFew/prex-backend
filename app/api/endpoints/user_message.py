from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, UUID4
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from app.tools.support_tools import (
    record_feature_request,
    record_feedback,
    search_kb,
    search_info_store,
    escalate_ticket,
    update_ticket_status,
    add_ticket_note,
    update_ticket_name
)
import json
import re

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

# Setup LLM for memory
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationSummaryMemory(
    llm=memory_llm,
    return_messages=True,
    max_token_limit=1000
)

# Setup LLM and agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support AI. Use your tools to help users.
    - If user asks about features, use record_feature_request
    - If user gives feedback, use record_feedback
    - Search KB and info store for relevant info
    - IMPORTANT: If you need to escalate to a human agent or mention escalation in your response, you MUST use the escalate_ticket tool first
    - ALWAYS use escalate_ticket tool BEFORE telling user you will escalate their request
    - IMPORTANT: When user indicates their issue is resolved (says things like "that fixed it", "all good now", "thanks for the help", etc), ALWAYS use update_ticket_status with status='closed'
    - When starting to help user with issue, use update_ticket_status with status='in_progress'
    - For fresh tickets, use update_ticket_name to set a descriptive title based on user's issue
    - Use add_ticket_note to record important information about:
        * Nature of the issue when first reported
        * Key details discovered during conversation
        * Steps taken to resolve the issue
        * Resolution details when issue is solved
    
    Keep responses friendly but professional."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [
    record_feature_request,
    record_feedback,
    search_kb,
    search_info_store,
    escalate_ticket,
    update_ticket_status,
    add_ticket_note,
    update_ticket_name
]

# Create agent with OpenAIFunctionsAgent
agent = OpenAIFunctionsAgent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

@router.post("/user-message", response_model=UserMessageResponse)
async def handle_user_message(request: UserMessageRequest):
    try:
        # Get latest message
        latest_message = request.messages[-1].message if request.messages else ""
        
        # Add messages to memory and get summary
        memory.clear()  # Clear previous memory
        for msg in request.messages[:-1]:  # Exclude latest message
            if msg.is_system_message:
                memory.chat_memory.add_ai_message(msg.message)
            else:
                memory.chat_memory.add_user_message(msg.message)
        
        # Get memory summary if we have history
        memory_summary = ""
        if len(request.messages) > 1:
            memory_messages = memory.chat_memory.messages
            if memory_messages:
                memory_summary = "\nContext from previous conversations:\n" + memory.predict_new_summary(
                    memory_messages, ""
                )
        
        # Format chat history for LangChain
        formatted_history = []
        for msg in request.messages[:-1]:  # Exclude latest message
            role = "assistant" if msg.is_system_message else "human"
            formatted_history.append((role, msg.message))
        
        # Run agent with memory summary
        result = await agent_executor.ainvoke({
            "input": latest_message,
            "chat_history": formatted_history,
            "memory_summary": memory_summary
        })

        print(f"Debug - full result: {result}")  # Debug full result

        # Get raw response text first
        response_text = str(result["output"])
        print(f"Debug - raw response: {response_text}")

        # Parse tool outputs from actions
        actions = []
        
        # Get actions from intermediate steps
        if "intermediate_steps" in result:
            steps = result["intermediate_steps"]
            print(f"Debug - intermediate steps: {steps}")
            
            for step in steps:
                # Handle AgentAction format
                if isinstance(step, tuple) and len(step) == 2:
                    tool_return = step[1]
                    print(f"Debug - tool return: {tool_return}")
                    
                    if isinstance(tool_return, dict) and "action" in tool_return:
                        print(f"Debug - found action: {tool_return}")
                        actions.append(tool_return)
                    elif isinstance(tool_return, str):
                        try:
                            # Try parse JSON from string
                            parsed = json.loads(tool_return.replace("'", '"'))
                            if isinstance(parsed, dict) and "action" in parsed:
                                print(f"Debug - parsed action: {parsed}")
                                actions.append(parsed)
                        except:
                            pass

        print(f"Debug - actions after steps: {actions}")

        # Clean up response text more aggressively
        cleaned_lines = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Skip lines that look like tool output
            if (line.startswith('{') or 
                line.startswith("'action':") or 
                line.startswith('"action":') or
                line.endswith('}') or
                'metadata' in line or
                'status' in line and ('closed' in line or 'in_progress' in line)):
                continue
                
            cleaned_lines.append(line)
        
        response_text = ' '.join(cleaned_lines).strip()
        
        # Check if we need to escalate
        escalate = any(a.get("action") == "escalate" for a in actions)
        
        # Get response text and check for escalation words
        escalation_words = ['escalate', 'escalated', 'human agent', 'support representative', 'real person']
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
            response=response_text,
            escalate=escalate,
            actions=actions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 