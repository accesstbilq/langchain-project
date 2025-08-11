from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import validators
import requests, traceback
import json
import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# Global variables for chat system and messages
chat_system = ""
messages = []

# Get the OpenAI API key from environment variables
OPENAIKEY = os.getenv('OPEN_AI_KEY')


#system_message = "You are a helpful assistant tasked with performing arithmetic on a set of inputs."

system_message = """You are an expert SEO assistant that can use tools to help with search engine optimization tasks. 
        
Your expertise includes:
- Keyword research and analysis
- Content optimization strategies
- Technical SEO recommendations
- Meta tag optimization
- Link building strategies
- SEO auditing and reporting
- Search ranking analysis
- Competitor analysis

You should provide actionable, data-driven SEO advice and use available tools when they can help gather information or perform specific SEO-related tasks. Always explain your recommendations clearly and provide practical implementation steps."""
        



@tool
def multiply(a: int, b: int) -> int:
        """Multiply a and b.
        Args:
        a: first int
        b: second int
        """
        chat_system = " Tool call "
        return a * b

def chatbot_view(request):
    return render(request, 'chatbot.html')

@csrf_exempt
def chatbot_input(request):
    """Enhanced chatbot view """        
    usermessage = request.POST.get('message', '').strip()
    # Add user message to conversation history
    #messages.append({"role": "user", "content": usermessage})

    try:
        tools = [multiply]
        llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAIKEY, model="gpt-3.5-turbo")
        #llm_with_tools = llm.bind_tools(tools,tools_choice="any")
        llm_with_tools = llm.bind_tools(tools)
        # Create a simple conversation
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}")
        ])
        
        chain = prompt | llm_with_tools
        chat_system = " LLM Call "
        default_response = chain.invoke({"input": usermessage})
        print (default_response)
        return JsonResponse({
                'success': True,
                'response': default_response.content,
                'session_id': "12345",
                'message_id': "12345",
                'response_meta': {
                    'source': chat_system,
                    'response_type': 'general_conversation',
                    'has_vectorstore': chat_system is not None
                }
            })
        
    except Exception as e:
        print(f"Error in default chat: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."
    
   