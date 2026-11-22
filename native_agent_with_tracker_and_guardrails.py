from __future__ import annotations

import json
import urllib.parse
import urllib.request
import ollama

# =====================================================================
# 1. TOKEN & COST TRACKER
# =====================================================================
class UsageTracker:
    def __init__(self, prompt_price_per_m: float = 0.0, eval_price_per_m: float = 0.0):
        """
        For local models (Ollama), prices are $0.0, but tracking tokens is vital
        for context window monitoring. Pass actual prices for commercial APIs.
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_price_per_m = prompt_price_per_m
        self.eval_price_per_m = eval_price_per_m

    def update(self, response):
        """Extract token metrics from Ollama response object."""
        # Ollama returns `prompt_eval_count` and `eval_count`
        prompt_count = getattr(response, 'prompt_eval_count', 0) or 0
        eval_count = getattr(response, 'eval_count', 0) or 0

        self.prompt_tokens += prompt_count
        self.completion_tokens += eval_count

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost(self) -> float:
        input_cost = (self.prompt_tokens / 1_000_000) * self.prompt_price_per_m
        output_cost = (self.completion_tokens / 1_000_000) * self.eval_price_per_m
        return input_cost + output_cost

    def summary(self) -> str:
        return (f"Tokens Used: {self.total_tokens} "
                f"(Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}) | "
                f"Est. Cost: ${self.estimated_cost:.5f}")


# =====================================================================
# 2. GUARDRAILS
# =====================================================================
ALLOWED_CITIES = {"tokyo", "tel aviv", "new york", "london", "paris", "kfar saba"}

def validate_input_prompt(prompt: str) -> bool:
    """Input Guardrail: Prevent huge payloads or malicious keywords.

    Demo only: a keyword blocklist is trivially bypassed. Real systems need
    layered defenses (moderation models, least-privilege tools, output checks).
    """
    if len(prompt) > 500:
        print("[Guardrail Blocked]: Prompt exceeds maximum allowable character limit.")
        return False

    forbidden_keywords = ["drop database", "ignore previous instructions", "system prompt"]
    if any(keyword in prompt.lower() for keyword in forbidden_keywords):
        print("[Guardrail Blocked]: Forbidden patterns detected in prompt.")
        return False

    return True

def validate_tool_args(tool_name: str, args: dict) -> bool:
    """Tool Execution Guardrail: Validate parameters before running Python code."""
    if tool_name == "get_live_weather":
        location = str(args.get("location") or "").strip().lower()
        if not location:
            print("[Tool Guardrail Error]: Location parameter is required.")
            return False
        # Example whitelist/sanitization policy
        if location not in ALLOWED_CITIES:
            print(f"[Tool Guardrail Blocked]: '{location}' is not in the allowed destinations list.")
            return False

    return True


# =====================================================================
# 3. REAL TOOL
# =====================================================================
def get_live_weather(location: str) -> str:
    """Fetch real-time weather information for an allowed city."""
    try:
        encoded_location = urllib.parse.quote(location)
        url = f"https://wttr.in/{encoded_location}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            current = data["current_condition"][0]
            return f"Current weather in {location}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}."
    except Exception as e:
        # Return the error as text so the LLM can see it and respond gracefully
        return f"Tool Execution Failure: {str(e)}"

AVAILABLE_TOOLS = {"get_live_weather": get_live_weather}


# =====================================================================
# 4. PROTECTED AGENT LOOP
# =====================================================================
def run_agent(user_prompt: str, model_name: str = "llama3.2:latest", max_iterations: int = 5) -> str | None:
    # STEP 1: INPUT GUARDRAIL CHECK
    if not validate_input_prompt(user_prompt):
        return None  # rejected by input guardrail

    # Initialize Token/Cost Tracker
    # Set pricing e.g. for OpenAI GPT-4o: prompt_price_per_m=2.50, eval_price_per_m=10.00
    tracker = UsageTracker(prompt_price_per_m=0.0, eval_price_per_m=0.0)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_prompt}
    ]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Call LLM
        response = ollama.chat(model=model_name, messages=messages, tools=[get_live_weather])

        # TRACK USAGE METRICS
        tracker.update(response)
        print(f"[Metrics]: {tracker.summary()}")

        response_message = response.message
        messages.append(response_message)

        tool_calls = response_message.tool_calls

        # DECIDE: Terminal state if no tool requested
        if not tool_calls:
            print("\n[Final Answer]:", response_message.content)
            return response_message.content

        # ACT: Execute tools with Guardrails
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            print(f"[Tool Requested]: `{function_name}` with args {function_args}")

            # STEP 2: TOOL GUARDRAIL CHECK
            if not validate_tool_args(function_name, function_args):
                tool_output = "Error: Tool execution denied by security policy."
            elif function_name in AVAILABLE_TOOLS:
                tool_output = AVAILABLE_TOOLS[function_name](**function_args)
            else:
                tool_output = f"Error: Tool '{function_name}' not registered."

            print(f"[Tool Execution Result]: {tool_output}")

            # Append output back to context
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": str(tool_output)
            })

    print("[Warning]: Loop limit exceeded.")
    return None  # no final answer within max_iterations


if __name__ == "__main__":
    # Test 1: Valid city
    print("=== TEST 1: Valid Query ===")
    run_agent("What is the weather right now in Tokyo?")

    # Test 2: Tool Guardrail Triggered
    print("\n=== TEST 2: Blocked City ===")
    run_agent("What is the weather right now in Miami?")
