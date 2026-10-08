import json
import urllib.parse
import urllib.request
import ollama

# 1. Real Tool Implementation: Fetching live data from wttr.in JSON endpoint
def get_live_weather(location: str) -> str:
    """
    Fetch real-time weather information for any given city.

    Args:
        location: The city or region name, e.g. "Tel Aviv", "Tokyo", "Kfar Saba"
    """
    try:
        # Encode city name for URL safety (e.g., spaces to %20)
        encoded_location = urllib.parse.quote(location)
        url = f"https://wttr.in/{encoded_location}?format=j1"
        
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return f"Error: Unable to reach weather service (status {response.status})."
            
            data = json.loads(response.read().decode("utf-8"))
            
            # Extract relevant fields from wttr.in JSON schema
            current = data["current_condition"][0]
            temp_c = current["temp_C"]
            condition = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            wind_kmh = current["windspeedKmph"]
            
            return f"Current weather in {location}: {temp_c}°C, {condition}. Humidity: {humidity}%, Wind: {wind_kmh} km/h."
            
    except Exception as e:
        return f"Failed to retrieve weather data for {location}: {str(e)}"


# Map schema name to executable python function
AVAILABLE_TOOLS = {
    "get_live_weather": get_live_weather
}


def run_agent(user_prompt: str, model_name: str = "llama3.2:latest", max_iterations: int = 5):
    """Agent execution loop using Ollama and live external API calls."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to real-time tools. Answer directly and use tools when required."
        },
        {"role": "user", "content": user_prompt}
    ]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # 1. PERCEIVE: Pass the live function as an executable tool
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=[get_live_weather]
        )

        response_message = response.message
        messages.append(response_message)

        tool_calls = response_message.tool_calls

        # 2. DECIDE: If no tools called, print the final LLM response
        if not tool_calls:
            print("\n[Agent Final Answer]:")
            print(response_message.content)
            return response_message.content

        # 3. ACT: Execute real API tool calls requested by model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            print(f"[Tool Call]: Invoking `{function_name}` with arguments: {function_args}")

            if function_name in AVAILABLE_TOOLS:
                tool_output = AVAILABLE_TOOLS[function_name](**function_args)
            else:
                tool_output = f"Error: Tool '{function_name}' is not registered."

            print(f"[Tool Output]: {tool_output}")

            # 4. FEEDBACK: Pass execution outcome back to context
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": str(tool_output)
            })

    print("[Warning]: Loop limit exceeded.")
    return "Agent timed out."


if __name__ == "__main__":
    # Test with live queries
    run_agent("What is the weather right now in Tokyo?")