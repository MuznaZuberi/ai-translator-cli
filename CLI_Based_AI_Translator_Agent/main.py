import os  # Used to interact with the operating system (e.g., to access environment variables)
from dotenv import load_dotenv  # Loads environment variables from a .env file
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
# Importing necessary classes from the 'agents' library to build and run the AI agent

load_dotenv()  # Load environment variables from the .env file into the system environment

gemini_api_key = os.getenv("GRMINI_API_KEY")  # Get the Gemini API key from the environment variables

# If the API key is not found, raise an error to avoid running without authentication
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Create an asynchronous OpenAI-style client configured to use the Google Gemini API endpoint
external_client = AsyncOpenAI(
    api_key=gemini_api_key,  # Use the API key loaded from .env
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini-compatible OpenAI-style endpoint
)

# Define the model to be used for chat completions, in this case Gemini 2.0 Flash
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  # The model name as per Gemini API documentation
    openai_client=external_client  # The client that communicates with Gemini
)

# Configuration for running the agent, includes the model and disables tracing
config = RunConfig(
    model=model,  # Model to use during the run
    model_provider=external_client,  # API client used to access the model
    tracing_disabled=True  # Disable tracing/logging of the run for cleaner execution
)

# Define an AI agent named "Translator" with specific instructions
mine_agent = Agent(
    name="Translator",  # Name of the agent
    instructions="You are a translator. Always translate Urdu sentences into fluent English."  # Agent's role and behavior
)

# Run the agent synchronously with a given Urdu input sentence using the defined configuration
res = Runner.run_sync(
    mine_agent,  # The agent that will handle the task
    input="مزنہ عامر زبیری — ڈویلپر | GIAIC کی طالبہ | ویب اور مصنوعی ذہانت کی ٹیکنالوجیز کی شوقین",  # Input in Urdu to be translated
    run_config=config  # Configuration including model and client details
)

print(res)  # Print the output (expected: translated sentence in fluent English)
