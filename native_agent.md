# Here is the native agent flow:
### --- INSIDE THE FOR LOOP (Iteration 1) ---

### 1. Line 63: You call the LLM for the first time.
response = ollama.chat(model=model_name, messages=messages, tools=[get_live_weather])

### 2. Line 88: Your script executes the Python function locally.
tool_output = AVAILABLE_TOOLS[function_name](**function_args)

### 3. Line 95: Your script appends the tool's result to your local array in memory.
    *** NO NETWORK / NO LLM CALL HAPPENS HERE ***
messages.append({
    "role": "tool",
    "name": function_name,
    "content": str(tool_output)
})

### --- LOOP RESTARTS (Iteration 2) ---

### 4. Line 63: NOW the script calls the LLM a second time!
   This time, `messages` contains the user question AND the tool result from line 95.
response = ollama.chat(model=model_name, messages=messages, tools=[get_live_weather])


```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Script as Python Script (Local Memory)
    participant LLM as Ollama / Llama 3.2
    participant API as External Weather API (wttr.in)

    User->>Script: "What is the weather in Tokyo?"
    Note over Script: Append user prompt to `messages` array

    rect rgb(240, 244, 250)
        note over Script, LLM: Iteration 1 — Decision Phase
        Script->>LLM: 1. ollama.chat(messages, tools=[get_live_weather])
        Note over LLM: Evaluates query + tool schemas.<br/>Decides tool call is needed.
        LLM-->>Script: 2. Response: tool_calls=[{name: "get_live_weather", args: {location: "Tokyo"}}]
        Note over Script: Append LLM's tool call request to `messages`
    end

    rect rgb(245, 250, 240)
        note over Script, API: Execution Phase (Local Python)
        Script->>API: HTTP GET [https://wttr.in/Tokyo?format=j1](https://wttr.in/Tokyo?format=j1)
        API-->>Script: Returns JSON: { temp_C: "19", condition: "Sunny", ... }
        Note over Script: Format JSON output string.<br/>Line 95: Append tool output to `messages` array.<br/>(NO LLM call yet)
    end

    rect rgb(240, 244, 250)
        note over Script, LLM: Iteration 2 — Synthesis Phase
        Script->>LLM: 3. ollama.chat(updated_messages)
        Note over LLM: Sees tool result in context.<br/>Determines task is complete.
        LLM-->>Script: 4. Response: content="The weather in Tokyo is 19°C...", tool_calls=None
        Note over Script: `tool_calls` is None → Exit agent loop
    end

    Script-->>User: "The weather right now in Tokyo is 19°C and sunny..."
