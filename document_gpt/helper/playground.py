from langchain.agents import Tool
from langchain.chat_models.openai import ChatOpenAI
from langchain.llms.openai import BaseOpenAI, OpenAI, OpenAIChat
from typing import List, Union
import re

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.vectorstores.chroma import Chroma


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

    # custom output parser


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output, provide default values
        if not match:
            action = "Default Action"
            action_input = "Default Action Input"
        else:
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"')

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


output_parser = CustomOutputParser()


embeddings = OpenAIEmbeddings()
chroma = Chroma(
    persist_directory='data/db',
    embedding_function=embeddings
)


# chat completion llm
# llm = ChatOpenAI(
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
# )
# llm = BaseOpenAI(base_url="https://api.kwwai.top/v1",
#                  api_key="sk-WjWNNIQN23IxAAHP2e3444539140409290A6E73661F3Ef0d")
llm = ChatOpenAI()
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=0,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma.as_retriever()
)

search = DuckDuckGoSearchRun()

tools = [
    Tool(name="Knowledge Base",
         description="Usefuk when answering by querying the document.", func=qa.run),
    Tool(name="search", description="Useful when answering questions that is not based on the document", func=search.run)
]
tool_names = [tool.name for tool in tools]

template = """
    You are a conversational AI, you respond everything you are told to, also you are a professional search engine who provides informative answers to users. Answer the following questions as best you can. You have access to the following tools:    {tools}

    Use the following format: 
    Question: the input question you must answer
    Thought: you should always think what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin! Remember to give detailed, informative answers
    
    Previous conversation history:
    {history}

    Question: {input}
    {agent_scratchpad}
    """


prompt = CustomPromptTemplate(template=template,
                              tools=tools,
                              input_variables=["input", "intermediate_steps", "history"])

agent_instructions = "Try 'pdf' tool first, Use the other tools if these don't work."

llm_chain = LLMChain(llm=llm, prompt=prompt)


output_parser = CustomOutputParser()

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

memory = ConversationBufferWindowMemory(k=3)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, hande_parsing_errors=True)

print(agent_executor.invoke(input("Enter your message: ")))
