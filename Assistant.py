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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

class LocalAssistant:
    store = {}
    def __init__(self) -> None:
        self.assistatn_Llm = None
        self.assistant_Template = None
        self.assistant_Parser = None
        self.assistant_retriver = None

    def create_Prompt_template(self,system_template):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", "{input}"),
            ]
        )
        self.assistant_Template = prompt
        print('prompt created')
        return prompt

    def create_LLM(self,llm_name = 'gpt-3.5-turbo'):
        model = ChatOpenAI(model= llm_name)
        self.assistatn_Llm = model
        print('LLm created')
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
    
    def load_docs(self, directory):
        docs = []
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                loader = PyPDFLoader(pdf_path)
                document = loader.load()
                docs.append(document)
        print('load documents')
        return docs
    
    def split(self,docs,chunk_size = 1000, chunk_overlap = 200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for doc in docs:
            splits = text_splitter.split_documents(doc)
        print('split documents')
        return splits
    
    def embed_store(self,splits,p_directory):
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=p_directory) 
        print(f'embed and save to {p_directory}')
        return vectorstore
    
    def text_to_index(self,directory_path,p_directory = "./index", embeding_model = "text-embedding-3-large"):
        docs = self.load_docs(directory=directory_path)
        splits = self.split(docs=docs)
        VS = self.embed_store(splits,p_directory = p_directory)
        return VS
       
    def create_retriver(self,p_directory = "./index",type_search = "similarity",top_k = 5):
        vectordb = Chroma(persist_directory=p_directory, embedding_function=OpenAIEmbeddings())
        retriever = vectordb.as_retriever(search_type=type_search, search_kwargs={"k": top_k})
        self.assistant_retriver = retriever
        return retriever
    
    def create_stuff_and_retrival_chain(self, input,llm,prompt,retriver):
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriver, question_answer_chain)
        results = rag_chain.invoke({"input": input})
        return results['answer']

    def create_parser(self):
        parser = StrOutputParser()
        self.assistant_Parser = parser
        return parser
    
    def chain_assitant(self,text , lang = "italian"):
        chain = self.assistant_Template | self.assistatn_Llm | self.assistant_Parser
        return chain.invoke({"language": lang, "text": text})


