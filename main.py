# Import necessary classes from the Agent SDK
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables (e.g., API keys) from the .env file
from dotenv import load_dotenv
import os
load_dotenv()

# Fetch Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
)

# Define the language model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",      
    openai_client=provider      
)

# Configuration object to control how the agent runs
config = RunConfig(
    model=model,                   # Use the custom model defined above
    model_provider=provider,      # Model provider configuration
    tracing_disabled=True         # Disables OpenAI telemetry tracing
)


# Create the agent with a simple instruction set
agent = Agent(
    name="Chatbot Assistant",
    instructions="You are a helpful chatbot assistant"
)

print("\nAssalam Walaikum, Welcome from Muneeb Lodhi\n")
while True:
    # Take user input from terminal
    user_input = input("Ask Freely Anything: ")

    if user_input.strip().lower() == "exit":
        print("\nGoodbye, Exiting the Agent.\n")
        break

    # Run the agent using synchronous execution
    res = Runner.run_sync(agent, user_input, run_config=config)

    # Output the final result
    print(res.final_output)
    print("\n")

