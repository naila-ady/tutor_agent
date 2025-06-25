import os
from agents import Agent
from agents import OpenAIChatCompletionsModel,AsyncOpenAI,Runner
from agents.run import RunConfig
from dotenv import load_dotenv,find_dotenv 

load_dotenv(find_dotenv())

gemini_api_key=os.getenv("GEMINI_API_KEY")

provider=AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model=OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = provider 
)

run_config=RunConfig(
    model = model,
    model_provider= provider,
    tracing_disabled = True
)


math_tutor_agent=Agent(
    name="Math Tutor",
    instructions="You are a Math instructor,You provide with Math problems"   
)

python_tutor_agent =Agent(
    name = "Ptyhon Tutor",
    handoff_description = "You are a special Python Agent",
    instructions = "you are a python instructor,solve the errors and answer the qusetions about Python Code"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question and always give short and summarized answers",
    handoffs=[python_tutor_agent, math_tutor_agent]
)

result = Runner.run_sync(triage_agent,"what is pydantic ",run_config=run_config)
print(result.final_output)

