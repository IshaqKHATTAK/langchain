from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from helper import system_template
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os

load_dotenv()
api_key =  os.getenv('OPENAI_API')
os.environ["OPENAI_API_KEY"] = api_key


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

model = ChatOpenAI(model='gpt-3.5-turbo')

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = prompt | model

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc5"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)
print("response after = ",response.content)

response = with_message_history.invoke(
    [HumanMessage(content="what is my name")],
    config=config,
)
print("response after = ",response.content)
#response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

#print("response after = ",response.content)
# parser = StrOutputParser()

# chain = prompt_template | model | parser
# answer = chain.invoke({"language": "italian", "text": "how can you help me?"})

# print("chain = ",answer)
