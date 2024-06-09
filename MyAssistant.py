from Assistant import LocalAssistant
from dotenv import load_dotenv
import os
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

# Question: {question} 

# Context: {context} 
#'''You are a helpful assistant. Answer all questions to the best of your ability.'''

# Answer:
sys_prompt = ''' "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    conext: {context} '''

load_dotenv()
api_key =  os.getenv('OPENAI_API')
os.environ["OPENAI_API_KEY"] = api_key

My_assistant = LocalAssistant()
model = My_assistant.create_LLM('gpt-3.5-turbo')
prompt = My_assistant.create_Prompt_template(sys_prompt)

My_assistant.text_to_index('./data/',"./index",embeding_model="text-mebedding-3-large")

retriever = My_assistant.create_retriver()
while True:
    text = input('Enter a question')
    if text == 'quit':
        break
    output = My_assistant.create_stuff_and_retrival_chain(input= text,llm=model, prompt=prompt, retriver=retriever)
    print(output)

# output = My_assistant.chat_with_history("My name is ishaq")
# print(output)

# output = My_assistant.chat_with_history("what is my name")
# print(output)

# print('len of docs', My_assistant.text_to_index('./data/'))