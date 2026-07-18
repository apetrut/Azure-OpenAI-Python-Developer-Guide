# Multi-agent systems with LangGraph

Earlier in this guide, a single LangChain agent was created and armed with tools to perform vector lookups and concrete document id lookups via function calling.
That agent was then wrapped in a backend API and connected to a chat user interface.
A single agent works well when the scope of the assistant is narrow, but as an application grows to cover many distinct tasks, a single agent with a large collection of tools becomes harder to reason about, prompt, and maintain.

[LangGraph](https://langchain-ai.github.io/langgraph/) is an extension of the LangChain ecosystem designed for exactly this situation.
Where LangChain excels at composing chains and building an individual agent, LangGraph provides a framework for building **stateful, multi-agent** applications as a graph of nodes and edges.
Each node can be an agent, a tool, or a plain Python function, and the edges describe how control and state flow between them.
Because the graph is explicit, LangGraph makes it possible to build reliable, long-running, multi-turn experiences where multiple specialized agents collaborate to fulfill a request.

This chapter introduces the core LangGraph concepts and shows how they build on the RAG and agent patterns already covered in this guide.
It concludes by pointing to two complete, end-to-end workshops that build multi-agent applications backed by Azure Cosmos DB.

## From a single agent to a multi-agent system

In the [LangChain](../10_LangChain/README.md) chapter, a single agent decided which tool to call to answer a question about Cosmic Works products, customers, and sales orders.
A multi-agent system takes this idea further by dividing responsibilities across several **specialized agents**, each with its own prompt and its own focused set of tools.
A **coordinator** (or router) agent receives the incoming request and decides which specialized agent should handle it, transferring control as the conversation evolves.

This design has several advantages:

- **Separation of concerns** - each agent has a smaller, clearer prompt and a smaller set of tools, which improves reliability and makes the system easier to test.
- **Composability** - new capabilities are added by introducing a new agent and a transfer edge, rather than by growing a single prompt indefinitely.
- **Maintainability** - agents can be developed, evaluated, and updated independently.

> **Note**: A multi-agent system is not always the right choice. Start with a single agent, and introduce additional agents only when the range of tasks or the size of the prompt makes a single agent difficult to manage.

## Core LangGraph concepts

LangGraph introduces a small number of building blocks that work together to form an application:

- **State** - a shared object (commonly `MessagesState`, which holds the running list of messages) that flows through the graph and is updated by each node.
- **Nodes** - units of work in the graph. A node can wrap an agent, call a tool, or run arbitrary Python.
- **Edges** - connections that determine which node runs next. Edges can be fixed or conditional.
- **Agents** - created with `create_react_agent`, which builds a ReAct-style agent from a model, a set of tools, and a prompt.
- **Tools** - Python functions decorated with `@tool` that an agent can call, exactly like the tools introduced in the LangChain chapter.
- **Checkpointer** - persistence for the graph's state, which enables multi-turn conversations, durability, and memory. Azure Cosmos DB is an ideal store for these checkpoints.

## Creating specialized agents

Each specialized agent is created with `create_react_agent`, providing the language model, the tools that agent is allowed to use, and a prompt describing its role.

```python
from langgraph.prebuilt import create_react_agent

transactions_agent = create_react_agent(
    model,
    transactions_agent_tools,
    prompt=load_prompt("transactions_agent"),
)

sales_agent = create_react_agent(
    model,
    sales_agent_tools,
    prompt=load_prompt("sales_agent"),
)
```

## Coordinating agents with handoff tools

The coordinator agent is given a special kind of tool that allows it to **transfer** the conversation to another agent.
Handoff is modeled as just another tool call, which keeps the pattern consistent with everything already learned about tools.

```python
coordinator_agent_tools = [
    create_agent_transfer(agent_name="customer_support_agent"),
    create_agent_transfer(agent_name="transactions_agent"),
    create_agent_transfer(agent_name="sales_agent"),
]

coordinator_agent = create_react_agent(
    model,
    tools=coordinator_agent_tools,
    prompt=load_prompt("coordinator_agent"),
)
```

The coordinator's prompt describes when to route to each agent, for example: transfer to `sales_agent` to open a new account, or transfer to `transactions_agent` to check a balance or make a transfer.

## Composing the graph

Each agent is wrapped in a calling function and added to a `StateGraph` as a node.
A dedicated `human` node uses `interrupt` to pause the graph and collect the next user input, which is what enables natural multi-turn conversations.

```python
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import interrupt

def human_node(state: MessagesState, config) -> None:
    """A node for collecting user input."""
    interrupt(value="Ready for user input.")
    return None

builder = StateGraph(MessagesState)
builder.add_node("coordinator_agent", call_coordinator_agent)
builder.add_node("customer_support_agent", call_customer_support_agent)
builder.add_node("transactions_agent", call_transactions_agent)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("human", human_node)
builder.add_edge(START, "coordinator_agent")
```

## Persisting state and memory with Azure Cosmos DB

A distinguishing feature of LangGraph is its **checkpointer**, which persists the graph's state after each step.
By using Azure Cosmos DB as the checkpoint store, the same database that holds operational data and vectors also holds conversation state, so a conversation can span many turns, survive restarts, and be resumed later - without introducing a separate store to synchronize.

```python
from langchain_azure_cosmosdb import CosmosDBSaver

checkpointer = CosmosDBSaver(database_name=DATABASE_NAME, container_name="Checkpoints")
graph = builder.compile(checkpointer=checkpointer)
```

Once compiled, the graph is driven by streaming updates, which surface each agent's response as control moves through the graph:

```python
for update in graph.stream(input_message, config=thread_config, stream_mode="updates"):
    for node_id, value in update.items():
        if isinstance(value, dict) and value.get("messages"):
            last_message = value["messages"][-1]
            print(f"{node_id}: {last_message.content}")
```

Beyond raw checkpoints, agents often benefit from longer-term **memory** - remembering user preferences and summarizing past interactions across sessions.
The [azure-cosmos-agent-memory](https://pypi.org/project/azure-cosmos-agent-memory/) toolkit builds on this idea, automatically creating and managing memory containers in Azure Cosmos DB and handling fact extraction and summarization on your behalf.

## Go deeper: end-to-end multi-agent workshops

The concepts above are demonstrated in full in two complete, hands-on workshops that build multi-agent applications on Azure Cosmos DB using LangGraph, Azure OpenAI, a FastAPI backend, and an Angular front end.
Both are excellent next steps once you have completed this guide.

- **[Banking multi-agent workshop (Python, LangGraph)](https://github.com/AzureCosmosDB/banking-multi-agent-workshop/tree/WorkShop_v2_PythonLangGraph)** - build a multi-tenant retail banking assistant with a coordinator agent that routes to specialized customer support, transactions, and sales agents. Demonstrates agent handoff, tool calling, vector search over an offers container, and Azure Cosmos DB checkpointing.
- **[Travel multi-agent workshop (Python, LangGraph)](https://github.com/AzureCosmosDB/travel-multi-agent-workshop/tree/agent_memory_toolkit_v2)** - build a travel-planning assistant with specialized hotel, dining, and activity agents coordinated by an orchestrator. Demonstrates the `azure-cosmos-agent-memory` toolkit for persistent agent memory, a Model Context Protocol (MCP) server, and deployment with the Azure Developer CLI (`azd up`).

> **Note**: These workshops are maintained independently of this guide and evolve quickly. Refer to each workshop's own README for the current prerequisites, package versions, and deployment steps.

## Summary

LangGraph builds directly on the LangChain and RAG patterns from earlier in this guide, adding an explicit graph structure, durable state, and first-class support for multiple collaborating agents.
By persisting checkpoints and memory in Azure Cosmos DB, a multi-agent application keeps its operational data, vectors, and conversation state together in a single database - the same unified-data advantage highlighted throughout this guide, now extended to agentic workloads.
