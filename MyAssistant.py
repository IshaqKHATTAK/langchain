from Assistant import LocalAssistant
from dotenv import load_dotenv
import os


load_dotenv()
api_key =  os.getenv('OPENAI_API')
os.environ["OPENAI_API_KEY"] = api_key

My_assistant = LocalAssistant()
My_assistant.create_LLM('gpt-3.5-turbo')
My_assistant.create_Prompt_template('''You are a helpful assistant. Answer all questions to the best of your ability.''')

output = My_assistant.chat_with_history("My name is ishaq")
print(output)


output = My_assistant.chat_with_history("what is my name")
print(output)

#print('len of docs', My_assistant.text_to_index('./data/'))