from django.shortcuts import render
from django.http import JsonResponse
from .multi_parser import MultiURLVectorizer
from langchain.tools import tool
import validators
import requests
from bs4 import BeautifulSoup
from langchain.memory import ConversationVectorStoreTokenBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from typing import List, Dict, Any
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
import re
from django.views.decorators.csrf import csrf_exempt

chat_system = None
messages = []

# ------------------- Configuration -------------------

# Load ENV file
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAIKEY = os.getenv('OPEN_AI_KEY')

# ------------------- Tools -------------------

# Define custom tools using LangChain's @tool decorator
@tool("validate_and_fetch_url", return_direct=True)
def validate_and_fetch_url(url: str) -> str:
    """Validate a URL and fetch its title if valid."""
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        
        return f"✅ URL is valid.\nTitle: {title}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ URL validation passed, but content fetch failed.\nError: {str(e)}"

# ------------------- Views -------------------

def chatbot_view(request):
    return render(request, 'chatbot/chat.html')

# Enhanced view specifically for URL validation

@csrf_exempt
def validate_url_view(request):
    """Dedicated view for URL validation"""
    if request.method == 'POST':
        try:

            usermessage = request.POST.get('message', '').strip()
            
            messages.append({"role": "user", "content": usermessage})

            url_input = "https://www.brihaspatitech.com/ecommerce-development-company/"    

            vectorizer = MultiURLVectorizer(
                urls=[url_input],
                embedding_model="openai",
                chunk_size=1000,
                delay_between_requests=0.5
            )
            summary = vectorizer.process(parallel=False)
            
            if summary["successful_urls"]:
                # Initialize chat system
                chat_system = EnhancedWebContentChat(vectorizer)
                
                

                # Display summary
                url_summary = summary["processing_details"][url_input]
                tool_response = chat_system.get_tool_response(usermessage)
                response = chat_system.get_response(usermessage)

                
                print(tool_response)  # Output: Valid URL ✅

                if tool_response["success"] == True :
                    assistant_message = {
                        "role": "assistant", 
                        "content": tool_response["answer"]["text"],
                        "sources": 0
                    }
                else:
                    assistant_message = {
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["source_documents"]
                    }

                



                messages.append(assistant_message)

                # Memory management
                print(messages)

                #get Chat history
                if chat_system:
                    history = chat_system.memory_manager.get_chat_history()

                print(history)

                if tool_response["success"]:
                    return JsonResponse({
                        'success': True,
                        'response': tool_response["answer"]["text"]
                    })
                else:
                    return JsonResponse({
                        'success': True,
                        'response': response["answer"]
                    })


            else:
                return JsonResponse({
                    'success': False,
                    'response': '❌ Failed to process the URL.',
                    'processing_details': summary["processing_details"][url_input]
                })
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'response': 'Invalid JSON data provided.'
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'response': f'Error validating URL: {str(e)}'
            })
        
    return JsonResponse({
        'success': False,
        'response': 'Only POST requests are allowed.'
    })

# Your existing classes remain the same
class ChatMemoryManager:
    """Manages chat history and memory for conversational interactions"""
    
    def __init__(self, vectorstore, window_token_limit: int = 1000):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        self.memory = ConversationVectorStoreTokenBufferMemory(
            retriever=retriever,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            llm=ChatOpenAI(temperature=0, openai_api_key=OPENAIKEY),
            max_token_limit=window_token_limit,
        )
        self.llm = self.memory.llm
        
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history for display"""
        history = []
        if hasattr(self.memory, 'chat_memory'):
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "human", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "ai", "content": message.content})
        return history
    
    def add_message(self, human_input: str, ai_response: str):
        """Add a conversation pair to memory"""
        self.memory.save_context(
            {"input": human_input},
            {"answer": ai_response}
        )
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()

class EnhancedWebContentChat:
    """Enhanced web content chat with memory and context awareness"""
    
    def __init__(self, vectorizer: MultiURLVectorizer = None):
        self.vectorizer = vectorizer
        self.memory_manager = None
        if vectorizer and vectorizer.vectorstore:
            self.memory_manager = ChatMemoryManager(vectorstore=vectorizer.vectorstore)
        
        self.system_prompt = """You are a helpful AI assistant that answers questions based on web content that has been processed and stored in a vector database. 

Context about the processed content:
- The content comes from web pages that have been parsed and chunked
- You have access to metadata including titles, headings, FAQs, and main content
- Each piece of content has source URL information

Instructions:
- Always provide accurate information based on the retrieved content
- When referencing information, mention the source URL when relevant
- If you don't have enough information to answer a question, say so clearly
- Maintain context from previous conversations
- Be conversational and helpful
- If asked about previous conversations, refer to the chat history appropriately

Current time: {current_time}
"""

        self.contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Based on the following context, please answer my question:\n\nContext: {context}\n\nQuestion: {input}")
        ])

    def setup_retrieval_chain(self):
        """Setup the retrieval chain with memory"""
        if not self.vectorizer or not self.vectorizer.vectorstore:
            raise ValueError("Vector store not available. Process URLs first.")
        
        if not self.memory_manager:
            self.memory_manager = ChatMemoryManager(vectorstore=self.vectorizer.vectorstore)
        
        retriever = self.vectorizer.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 8}
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.memory_manager.llm,
            retriever,
            self.contextualize_q_prompt
        )
        
        question_answer_chain = create_stuff_documents_chain(
            self.memory_manager.llm,
            self.qa_prompt
        )
        
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

    def setup_tool_agent(self):
        """Setup an agent that uses tools along with memory"""
        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")

        tools = [validate_and_fetch_url]

        # Create an agent that can use tools + memory
        self.tool_agent = initialize_agent(
            tools=tools,
            llm=self.memory_manager.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory_manager.memory,
            verbose=True
        )

    def get_tool_response(self, user_input: str) -> Dict[str, Any]:
        """Get response using tool agent with memory"""
        try:
            if not hasattr(self, 'tool_agent'):
                self.setup_tool_agent()

            result = self.tool_agent.invoke(user_input)

            print('nishant', result)

            return {
                "answer": {
                    "text": result
                },
                "success": True
            }


        except Exception as e:
            return {
                "answer": {
                    "text": f"Tool error: {str(e)}"
                },
                "success": False,
                "error": str(e)
            }

    def get_response(self, user_input: str) -> Dict[str, Any]:
        """Get response with memory context"""
        try:
            if not hasattr(self, 'rag_chain'):
                self.setup_retrieval_chain()
            
            chat_history = []
            if self.memory_manager and hasattr(self.memory_manager.memory, 'chat_memory'):
                chat_history = self.memory_manager.memory.chat_memory.messages
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            formatted_system_prompt = self.system_prompt.format(current_time=current_time)
            
            qa_prompt_with_time = ChatPromptTemplate.from_messages([
                ("system", formatted_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "Based on the following context, please answer my question:\n\nContext: {context}\n\nQuestion: {input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(
                self.memory_manager.llm,
                qa_prompt_with_time
            )
            
            retriever = self.vectorizer.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 8}
            )
            
            history_aware_retriever = create_history_aware_retriever(
                self.memory_manager.llm,
                retriever,
                self.contextualize_q_prompt
            )
            
            updated_rag_chain = create_retrieval_chain(
                history_aware_retriever,
                question_answer_chain
            )
            
            response = updated_rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            if self.memory_manager:
                self.memory_manager.add_message(user_input, response["answer"])
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("context", []),
                "success": True
            }
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            if self.memory_manager:
                self.memory_manager.add_message(user_input, error_message)
            return {
                "answer": error_message,
                "source_documents": [],
                "success": False,
                "error": str(e)
            }