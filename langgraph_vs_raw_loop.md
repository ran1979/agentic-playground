```mermaid
sequenceDiagram
    autonumber
    actor User
    participant State as AgentState (Memory)
    participant NodeAgent as Node: "agent" (LLM)
    participant EdgeCond as Edge: tools_condition
    participant NodeTools as Node: "tools" (ToolNode)

    User->>State: Input initial message
    State->>NodeAgent: Pass state["messages"]
    
    rect rgb(240, 244, 250)
        note over NodeAgent, EdgeCond: Iteration 1
        NodeAgent->>NodeAgent: ChatOllama.invoke()
        NodeAgent->>State: Append AIMessage(tool_calls=[...])
        State->>EdgeCond: Check last message
        EdgeCond-->>NodeTools: Route to "tools" (tool_calls present)
    end

    rect rgb(245, 250, 240)
        note over NodeTools, State: Execution
        NodeTools->>NodeTools: Execute get_live_weather("Tokyo")
        NodeTools->>State: Append ToolMessage(content="19°C, Sunny")
        State->>NodeAgent: Loop back edge ("tools" -> "agent")
    end

    rect rgb(240, 244, 250)
        note over NodeAgent, EdgeCond: Iteration 2
        NodeAgent->>NodeAgent: ChatOllama.invoke()
        NodeAgent->>State: Append AIMessage(content="The weather in Tokyo is...")
        State->>EdgeCond: Check last message
        EdgeCond-->>User: Route to END (no tool_calls)
    end