from langchain.llms import openai
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools.python import PythonREPL

llm = openai.OpenAI(temperature=0)

tools = [
    Tool(name="ExecCode", func=PythonREPL().run, description="Executa código")
]

## Agente que pensa e age com base na descrição das ferramentas
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("Calculate 2+2 in Python")