import os
from agents.run import RunConfig
from agents import( 
Agent, OpenAIChatCompletionsModel,AsyncOpenAI,Runner)
from dotenv import load_dotenv,find_dotenv 
from pydantic import BaseModel
import asyncio

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
class PythonHomeWorkOutput(BaseModel):
    is_Python_work : bool #boolean value represent the decision of work to be done
    reasoning : str #shows the reasons of the above decision y true or false
    answer:str #this is the answer of the question
    
guardrail_agent= Agent(
    name ="Guardrail Agent",
    instructions="Check if the user is askinng python language",
    output_type=PythonHomeWorkOutput,#we declaredhere the type of output by giving it class 
    model= model
    )

    
output = Runner.run_sync(guardrail_agent,"what is the string in Python?",run_config=run_config)
print(type(guardrail_agent.output_type))
print("is_python_work : ",output.final_output.is_Python_work)
print("Reasoning : ",output.final_output.reasoning)
print("Answer : ",output.final_output.answer)

