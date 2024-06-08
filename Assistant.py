from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from helper import system_template
from dotenv import load_dotenv
import os

class LocalAssistant:
    store = {}
    def __init__(self) -> None:
        self.assistatn_Llm = None
        self.assistant_Template = None
        self.assistant_Parser = None

    def create_Prompt_template(self,system_template = system_template):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                system_template,
                ),
            MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.assistant_Template = prompt
        return prompt

    def create_LLM(self,llm_name = 'gpt-3.5-turbo'):
        model = ChatOpenAI(model= llm_name)
        self.assistatn_Llm = model
        return model
    
    def chat_with_history(self,input,session_id = "abc5"):
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in LocalAssistant.store:
                LocalAssistant.store[session_id] = ChatMessageHistory()
                return LocalAssistant.store[session_id]
            
        chain = self.assistant_Template | self.assistatn_Llm
        with_message_history = RunnableWithMessageHistory(chain, get_session_history)
        config = {"configurable": {"session_id": session_id}}
        response = with_message_history.invoke(
            [HumanMessage(content=input)],
            config=config,
            )
        return response.content
    
    def text_to_index(self,directory_path):
        docs = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory_path, filename)
                loader = PyPDFLoader(pdf_path)
                document = loader.load()
                docs.append(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for doc in docs:
            splits = text_splitter.split_documents(doc)
    
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./index")
    def Retrive_And_Generate(self,input = "what is the main concept behind SSP?"):
        vectordb = Chroma(persist_directory="./index", embedding_function=OpenAIEmbeddings())
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        retrieved_docs = retriever.invoke(input)
        return retrieved_docs[0]

    
    def create_parser(self):
        parser = StrOutputParser()
        self.assistant_Parser = parser
        return parser
    
    def chain_assitant(self,text , lang = "italian"):
        chain = self.assistant_Template | self.assistatn_Llm | self.assistant_Parser
        return chain.invoke({"language": lang, "text": text})


