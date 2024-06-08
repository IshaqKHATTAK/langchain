from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os


load_dotenv()
openai_key = os.getenv('OPENAI_API')

os.environ['OPENAI_API_KEY'] = openai_key
model = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a fitness traniner help people stay helathy your name is EpoHealth trainer.''',
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "abc2"}}

chain = prompt | model
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

#give information and store in history
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm ishaq")],
    config=config,
)

print(response.content)

#check does it recall the history
response = with_message_history.invoke(
    [HumanMessage(content="what is my name?")],
    config=config,
)

print(response.content)