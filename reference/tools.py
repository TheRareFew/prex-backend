import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import matplotlib.pyplot as plt
import requests
import numpy as np

tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
  """Subtract b from a"""
  return a - b

@tool
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

@tool
def findAreaOfTrapazoid(top: int, bottom: int, height: int) -> int:
  """Calculate the area of a Trapazoid given the top, bottom and height"""
  return ((top + bottom) / 2) * height


@tool
def findAreaOfCircle(r: int) -> float:
  """Find the area of a circle using it's radius"""
  a = 3.14 * r ** 2
  return a

@tool
def display_circle(radius: int):
    """
    Displays a circle with the given radius using matplotlib.

    Args:
        radius (int): The radius of the circle to display.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a circle using the given radius
    circle = plt.Circle((0, 0), radius, fill=False, color='r', linewidth=2)

    # Add the circle to the axis
    ax.add_artist(circle)

    # Set the aspect ratio to be equal
    ax.set_aspect('equal')

    # Set the axis limits to show the entire circle
    ax.set_xlim((-radius - 1, radius + 1))
    ax.set_ylim((-radius - 1, radius + 1))

    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the circle
    plt.show()


    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a mathematical assistant. Use your tools to answer questions.
         If you do not have a tool to answer the question, say so.

        Return only the answers. e.g
        Human: What is 1 + 1?
        AI: 2
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
    
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Setup the toolkit
toolkit = [add, subtract, multiply, square, display_circle, findAreaOfCircle, findAreaOfTrapazoid]

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, toolkit, prompt)

# Create an agent executor by passing in the agent and toolkit
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

result = agent_executor.invoke({"input": "what is the area of a trapazoid given the height of 10, a top of 5 and a bottom of 12."})
print(result['output'])