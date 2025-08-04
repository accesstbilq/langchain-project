from django.shortcuts import render
from django.http import JsonResponse
from .multi_parser import MultiURLVectorizer
from langchain.tools import tool
import validators
import requests, traceback
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
    
@tool("web_scraper_tool", return_direct=True)
def web_scraper_tool(input_str: str) -> str:
    """
    Scrape content from a web page.

    Expected format: "https://example.com|text"
    Supported element types: text, headings, links, images, paragraphs, meta
    """
    try:
        # Split input string
        parts = input_str.strip().split("|")
        url = parts[0].strip()
        element_type = parts[1].strip().lower() if len(parts) > 1 else "text"
    except Exception:
        return "❌ Invalid input format. Use: <url>|<element_type> (e.g., https://example.com|text)"

    # Validate URL
    if not validators.url(url):
        return "❌ Invalid URL. Please provide a valid URL (e.g., https://example.com)."

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()

        result = f"🌐 Web Scraping Results from: {url}\n\n"

        if element_type == "text":
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            if len(text) > 2000:
                text = text
            result += f"📄 **Text Content:**\n{text}"

        elif element_type == "headings":
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            result += "📋 **Headings Found:**\n" + "\n".join(
                f"- {h.name.upper()}: {h.get_text().strip()}" for h in headings[:20]
            ) if headings else "📋 **Headings:** No headings found."

        elif element_type == "links":
            links = soup.find_all('a', href=True)
            unique_links = set()
            if links:
                result += "🔗 **Links Found:**\n"
                for link in links[:30]:
                    href = link['href']
                    text = link.get_text().strip()
                    if href not in unique_links and href.startswith(('http', 'https', '/')):
                        unique_links.add(href)
                        result += f"- {text} → {href}\n"
            else:
                result += "🔗 **Links:** No links found."

        elif element_type == "images":
            images = soup.find_all('img', src=True)
            result += "🖼️ **Images Found:**\n" + "\n".join(
                f"- {img.get('alt', 'No alt text')} → {img['src']}" for img in images[:20]
            ) if images else "🖼️ **Images:** No images found."

        elif element_type == "paragraphs":
            paragraphs = soup.find_all('p')
            result += "📝 **Paragraphs:**\n" + "\n\n".join(
                f"{i+1}. {p.get_text().strip()}" for i, p in enumerate(paragraphs[:10]) if p.get_text().strip()
            ) if paragraphs else "📝 **Paragraphs:** No paragraphs found."

        elif element_type == "meta":
            title = soup.find('title')
            description = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            keywords = soup.find('meta', attrs={'name': 'keywords'})
            result += "🏷️ **Meta Information:**\n"
            result += f"**Title:** {title.get_text() if title else 'No title found'}\n"
            result += f"**Description:** {description.get('content', 'No description') if description else 'No description found'}\n"
            result += f"**Keywords:** {keywords.get('content', 'No keywords') if keywords else 'No keywords found'}\n"

        else:
            return f"❌ Invalid element type '{element_type}'. Use: text, headings, links, images, paragraphs, meta."

        return result

    except requests.exceptions.RequestException as e:
        return f"⚠️ Failed to scrape the webpage.\nError: {str(e)}"
    except Exception as e:
        return f"❌ An error occurred during scraping.\nError: {str(e)}"
    
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
                

                if tool_response['success']:
                    assistant_message = {
                        "role": "assistant", 
                        "content": tool_response["answer"],
                        "sources": tool_response.get("sources", "")  # Add sources from tool response
                    }
                    messages.append(assistant_message)
                    
                    if tool_response['tool_used'] =='unknown':
                        chat_system.memory_manager.memory.output_key = "answer"
                        response = chat_system.get_response(usermessage)
                        
                        assistant_message = {
                            "role": "assistant", 
                            "content": response["answer"],
                            "sources": response.get("source_documents", "")
                        }
                        messages.append(assistant_message)

                        return JsonResponse({
                            'success': True,
                            'response': response["answer"],
                            'response_meta': {
                                'source': 'rag_chain',
                                'source_documents': len(response.get("source_documents", [])),
                                'url_processed': url_input,
                                'response_type': 'rag_response'
                            }
                        })

                    # Fixed: Include proper response_meta for tool responses
                    return JsonResponse({
                        'success': True,
                        'response': tool_response["answer"],
                        'response_meta': {
                            'source': 'tool_agent',
                            'tool_used': tool_response['tool_used'],  # Or detect which tool was used
                            'url_processed': url_input,
                            'response_type': 'direct_tool_response'
                        }
                    })
                else:
                    chat_system.memory_manager.memory.output_key = "answer"
                    response = chat_system.get_response(usermessage)
                    
                    assistant_message = {
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response.get("source_documents", "")
                    }
                    messages.append(assistant_message)

                    return JsonResponse({
                        'success': True,
                        'response': response["answer"],
                        'response_meta': {
                            'source': 'rag_chain',
                            'source_documents': len(response.get("source_documents", [])),
                            'url_processed': url_input,
                            'response_type': 'rag_response'
                        }
                    })

            else:
                return JsonResponse({
                    'success': False,
                    'response': '❌ Failed to process the URL.',
                    'processing_details': summary["processing_details"][url_input],
                    'response_meta': {
                        'source': 'error',
                        'error_type': 'url_processing_failed'
                    }
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'response': 'Invalid JSON data provided.',
                'response_meta': {
                    'source': 'error',
                    'error_type': 'json_decode_error'
                }
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'response': f'Error validating URL: {str(e)}',
                'response_meta': {
                    'source': 'error',
                    'error_type': 'general_exception',
                    'error_message': str(e)
                }
            })
        
    return JsonResponse({
        'success': False,
        'response': 'Only POST requests are allowed.',
        'response_meta': {
            'source': 'error',
            'error_type': 'invalid_method'
        }
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

        tools = [validate_and_fetch_url,web_scraper_tool]

        # Option 1: Fix the memory configuration
        print(f"DEBUG: Current memory type: {type(self.memory_manager.memory)}")
        
        # Check if memory has output_key attribute and fix it
        if hasattr(self.memory_manager.memory, 'output_key'):
            print(f"DEBUG: Current memory output_key: {self.memory_manager.memory.output_key}")
            # Change the memory's output_key to match what the agent produces
            self.memory_manager.memory.output_key = "output"
            print(f"DEBUG: Changed memory output_key to: {self.memory_manager.memory.output_key}")
        
        # Option 2: If the above doesn't work, create agent without memory and handle it manually
        try:
            self.tool_agent = initialize_agent(
                tools=tools,
                llm=self.memory_manager.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory_manager.memory,
                verbose=True,
                return_direct=False,
            )
            print("DEBUG: Agent created successfully with memory")
            self.use_manual_memory = False
            
        except Exception as e:
            print(f"DEBUG: Failed to create agent with memory: {e}")
            print("DEBUG: Creating agent without memory and will handle memory manually")
            
            self.tool_agent = initialize_agent(
                tools=tools,
                llm=self.memory_manager.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                return_direct=False,
                # No memory - we'll handle it manually
            )
            self.use_manual_memory = True
            print("DEBUG: Agent created without memory")

    def get_tool_response(self, user_input: str) -> Dict[str, Any]:
        """Get response using tool agent with memory"""
        try:
            if not hasattr(self, 'tool_agent'):
                self.setup_tool_agent()

            print(f"DEBUG: About to invoke agent with input: {user_input}")
            # Detect which tool might be used based on input
            tool_used = self._detect_tool_from_input(user_input)
            print("*" * 60)
            print('tool_used',tool_used)
            print("*" * 60)
            # The key fix: wrap the input in the correct format for the agent
            if isinstance(user_input, str):
                formatted_input = {"input": user_input}
            else:
                formatted_input = user_input
                
            result = self.tool_agent.invoke(formatted_input)
            
            print("DEBUG: Raw result from agent:")
            #print('nidshihdshkds', result)
            #print(f"DEBUG: Result type: {type(result)}")
            
            # Handle the response based on what the agent returns
            if isinstance(result, dict):
                print(f"DEBUG: Result keys: {list(result.keys())}")
                # Extract the response text from various possible keys
                response_text = (
                    result.get("output") or 
                    result.get("answer") or 
                    result.get("result") or 
                    str(result)
                )
            else:
                response_text = str(result)
            
            print(f"DEBUG: Extracted response_text: {response_text}")
            
            # Handle manual memory saving if needed
            if hasattr(self, 'use_manual_memory') and self.use_manual_memory:
                try:
                    # Try to save with the key your memory expects
                    if hasattr(self.memory_manager.memory, 'output_key'):
                        expected_key = self.memory_manager.memory.output_key
                    else:
                        expected_key = "answer"  # fallback to your original key
                        
                    self.memory_manager.memory.save_context(
                        {"input": user_input}, 
                        {expected_key: response_text}
                    )
                    print(f"DEBUG: Manually saved to memory with key: {expected_key}")
                except Exception as memory_error:
                    print(f"DEBUG: Manual memory save failed: {memory_error}")
            
            # Enhanced response with metadata
            print("*" * 60)
            print('tool_used',tool_used)
            print("*" * 60)

            final_result = {
                "answer": response_text,
                "success": True,
                "sources": self._extract_sources_from_response(response_text),
                "tool_used": tool_used,
                "agent_result": result,  # Include full agent result for debugging
                "metadata": {
                    "response_length": len(response_text),
                    "tool_detected": tool_used,
                    "memory_type": "manual" if hasattr(self, 'use_manual_memory') and self.use_manual_memory else "automatic"
                }
            }
            
            #print(f"DEBUG: Returning final result: {final_result}")
            return final_result

        except Exception as e:
            print(f"Exception occurred: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "answer": f"Tool error: {str(e)}",
                "success": False,
                "error": str(e),
                "sources": "",
                "tool_used": "unknown",
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_occurred": True
                }
            }

    def _extract_sources_from_response(self, response_text: str) -> str:
        """Extract source URLs or references from response text"""
        
        # Look for URLs in the response
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        urls = re.findall(url_pattern, response_text)
        
        if urls:
            return ", ".join(urls)
        
        # Look for other source indicators
        if "✅" in response_text and "URL is valid" in response_text:
            return "URL validation tool"
        elif "🌐 Web Scraping Results" in response_text:
            return "Web scraping tool"
        
        return ""
    def _detect_tool_from_input(self, user_input: str) -> str:
        """Detect which tool might be used based on user input"""
        input_lower = user_input.lower()
        print('Bansal',input_lower)
        if any(keyword in input_lower for keyword in ['valid', 'validate', 'check url', 'url valid']):
            return "validate_and_fetch_url"
        elif any(keyword in input_lower for keyword in ['scrape', 'extract', 'content', 'text', 'headings']):
            return "web_scraper_tool"
        else:
            return "unknown"


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


            print('Nishant ----- ', response)
            
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