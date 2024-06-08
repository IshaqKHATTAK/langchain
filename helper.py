system_template = '''You are a fitness traniner help people stay helathy your name is EpoHealth trainer.'''
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
openai_key = os.getenv('OPENAI_API')

os.environ['OPENAI_API_KEY'] = openai_key

model = ChatOpenAI(model='gpt-3.5-turbo')

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
chain = prompt_template | model | parser
#frist the input goes through the prompt template then goes throught the model then parser and parser gives the final answer
#note if we have variable in input then our input prompt and prompt template will change little
# {"var_name":"var_value","text": "whats your name!"}

print(chain.invoke({"text": "whats your name!"}))
