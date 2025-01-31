from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, UUID4
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from app.tools.support_tools import (
    record_feature_request,
    record_feedback,
    search_kb,
    search_info_store
)
from app.tools.manager_tools import (
    write_article,
    update_article_status,
    add_article_note
)

router = APIRouter()

class Message(BaseModel):
    message: str
    is_system_message: bool = False

class ManagerPromptRequest(BaseModel):
    prompt_id: UUID4
    conversation_id: UUID4
    prompt: str
    created_at: datetime
    chat_history: List[Message] = []

class ManagerPromptResponse(BaseModel):
    response: str
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
    ("system", """You are an AI manager assistant. Your role is to help managers make decisions and provide guidance.
    
    You can help managers with:
    1. Writing knowledge base articles based on their prompts
    2. Managing article statuses (draft, pending_approval, approved, rejected, archived)
    3. Adding notes to articles
    4. Recording feature requests and feedback
    5. Searching existing knowledge base and information store
    
    When writing articles:
    - Format content in HTML (not markdown)
    - Keep content professional and clear
    - Structure content logically with proper headings and paragraphs
    - Include relevant information from knowledge base if available
    - Do not include the article text in the response
    
    Keep responses professional and focused on management best practices."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [
    record_feature_request,
    record_feedback, 
    search_kb,
    search_info_store,
    write_article,
    update_article_status,
    add_article_note
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

@router.post("/manager-prompt", response_model=ManagerPromptResponse)
async def handle_manager_prompt(request: ManagerPromptRequest):
    try:
        # Add messages to memory and get summary
        memory.clear()  # Clear previous memory
        for msg in request.chat_history:
            if msg.is_system_message:
                memory.chat_memory.add_ai_message(msg.message)
            else:
                memory.chat_memory.add_user_message(msg.message)
        
        # Get memory summary if we have history
        memory_summary = ""
        if request.chat_history:
            memory_messages = memory.chat_memory.messages
            if memory_messages:
                memory_summary = "\nContext from previous conversations:\n" + memory.predict_new_summary(
                    memory_messages, ""
                )
        
        # Format chat history for LangChain
        formatted_history = []
        for msg in request.chat_history:
            role = "assistant" if msg.is_system_message else "human"
            formatted_history.append((role, msg.message))
        
        # Run agent with memory summary
        result = await agent_executor.ainvoke({
            "input": request.prompt,
            "chat_history": formatted_history,
            "memory_summary": memory_summary
        })

        print(f"Debug - full result: {result}")

        # Get raw response text
        response_text = str(result["output"])
        print(f"Debug - raw response: {response_text}")

        # Get actions from intermediate steps
        actions = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:  # Each step should have tool and output
                    tool_output = step[1]
                    if isinstance(tool_output, dict) and "action" in tool_output:
                        actions.append(tool_output)
                        print(f"Debug - adding action: {tool_output}")
        
        print(f"Debug - final actions: {actions}")
        
        return ManagerPromptResponse(
            response=response_text,
            actions=actions
        )
    except Exception as e:
        print(f"Error in handle_manager_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 