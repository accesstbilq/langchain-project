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

# Global variables for chat system and messages
chat_system = None
messages = []
current_vectorstore_url = None  # Track which URL the vectorstore was created for

# ------------------- Configuration -------------------

# Load ENV file
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAIKEY = os.getenv('OPEN_AI_KEY')

# ------------------- Tools -------------------

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
                text = text[:2000] + "..."
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

# ------------------- Helper Functions -------------------

def extract_url_from_message(message: str) -> str:
    """Extract URL from user message"""
    # Look for URLs in the message
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    urls = re.findall(url_pattern, message)
    
    # Return the first valid URL found
    for url in urls:
        if validators.url(url):
            return url
    
    return None

def create_default_chat_response(user_message: str) -> str:
    """Create a response using basic OpenAI chat without RAG"""
    try:
        llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAIKEY, model="gpt-3.5-turbo")
        
        # Create a simple conversation
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"input": user_message})
        
        return response.content
        
    except Exception as e:
        print(f"Error in default chat: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

# ------------------- Views -------------------

def chatbot_view(request):
    return render(request, 'chatbot/chat.html')

@csrf_exempt
def validate_url_view(request):
    """Enhanced chatbot view that handles URL processing, tools, RAG, and default chat"""
    global chat_system, messages, current_vectorstore_url
    
    if request.method == 'POST':
        try:
            usermessage = request.POST.get('message', '').strip()
            
            # Add user message to conversation history
            messages.append({"role": "user", "content": usermessage})

            # Step 1: Check if message contains a URL or is URL-related
            extracted_url = extract_url_from_message(usermessage)


            print('extracted_url ********************** ',extracted_url)
            
            # Step 2: If URL is found, process it and create/update vectorstore
            if extracted_url:
                print(f"URL detected: {extracted_url}")
                
                # Check if we need to create a new vectorstore for this URL
                if current_vectorstore_url != extracted_url:
                    print(f"Creating new vectorstore for URL: {extracted_url}")
                    
                    try:
                        # Create new vectorizer for the URL
                        vectorizer = MultiURLVectorizer(
                            urls=[extracted_url],
                            embedding_model="openai",
                            chunk_size=1000,
                            delay_between_requests=0.5
                        )
                        
                        summary = vectorizer.process(parallel=False)
                        
                        if summary["successful_urls"]:
                            # Create new chat system with this vectorstore
                            chat_system = EnhancedWebContentChat(vectorizer)
                            current_vectorstore_url = extracted_url
                            
                            # Provide initial analysis of the URL
                            url_info = f"✅ Successfully processed and analyzed: {extracted_url}\n\n"
                            url_info += f"📊 Processing Summary:\n"
                            url_info += f"- Documents created: {summary.get('total_documents_created', 0)}\n"
                            url_info += f"- Content analyzed: ✓\n\n"
                            url_info += "🤖 I'm now ready to answer questions about this website's content. What would you like to know?"
                            
                            messages.append({"role": "assistant", "content": url_info})
                            
                            return JsonResponse({
                                'success': True,
                                'response': url_info,
                                'response_meta': {
                                    'source': 'url_processing',
                                    'url_processed': extracted_url,
                                    'documents_created': summary.get('total_documents_created', 0),
                                    'response_type': 'vectorstore_created'
                                }
                            })
                        else:
                            error_msg = f"❌ Failed to process the URL: {extracted_url}"
                            messages.append({"role": "assistant", "content": error_msg})
                            
                            return JsonResponse({
                                'success': False,
                                'response': error_msg,
                                'response_meta': {
                                    'source': 'error',
                                    'error_type': 'url_processing_failed',
                                    'url': extracted_url
                                }
                            })
                            
                    except Exception as e:
                        error_msg = f"❌ Error processing URL {extracted_url}: {str(e)}"
                        messages.append({"role": "assistant", "content": error_msg})
                        
                        return JsonResponse({
                            'success': False,
                            'response': error_msg,
                            'response_meta': {
                                'source': 'error',
                                'error_type': 'url_processing_exception',
                                'url': extracted_url,
                                'error_message': str(e)
                            }
                        })

            # Step 3: If we have a chat system (URL was processed), use it
            if chat_system is not None:
                print("Using enhanced chat system with vectorstore...")
                
                # Try tool response first
                tool_response = chat_system.get_tool_response(usermessage)
                
                if tool_response['success'] and tool_response['tool_used'] != 'unknown':
                    assistant_message = {
                        "role": "assistant", 
                        "content": tool_response["answer"],
                        "sources": tool_response.get("sources", "")
                    }
                    messages.append(assistant_message)
                    
                    return JsonResponse({
                        'success': True,
                        'response': tool_response["answer"],
                        'response_meta': {
                            'source': 'tool_agent',
                            'tool_used': tool_response['tool_used'],
                            'url_processed': current_vectorstore_url,
                            'response_type': 'tool_response'
                        }
                    })
                
                # Use RAG system if no tool was used
                print("Using RAG system...")
                
                if hasattr(chat_system.memory_manager.memory, 'output_key'):
                    chat_system.memory_manager.memory.output_key = "answer"
                
                rag_response = chat_system.get_response(usermessage)
                
                if rag_response['success']:
                    assistant_message = {
                        "role": "assistant", 
                        "content": rag_response["answer"],
                        "sources": rag_response.get("source_documents", "")
                    }
                    messages.append(assistant_message)

                    return JsonResponse({
                        'success': True,
                        'response': rag_response["answer"],
                        'response_meta': {
                            'source': 'rag_chain',
                            'source_documents': len(rag_response.get("source_documents", [])),
                            'url_processed': current_vectorstore_url,
                            'response_type': 'rag_response'
                        }
                    })

            # Step 4: If no vectorstore and not URL-related, use default chat
            print("Using default chat mode...")
            
            default_response = create_default_chat_response(usermessage)
            
            assistant_message = {
                "role": "assistant", 
                "content": default_response
            }
            messages.append(assistant_message)
            
            return JsonResponse({
                'success': True,
                'response': default_response,
                'response_meta': {
                    'source': 'default_chat',
                    'response_type': 'general_conversation',
                    'has_vectorstore': chat_system is not None
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
            print(f"Exception in validate_url_view: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            
            return JsonResponse({
                'success': False,
                'response': f'Error processing request: {str(e)}',
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


# ------------------- Enhanced Classes (Keep existing classes unchanged) -------------------

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
            output_key="answer",  # Default to "answer"
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
        try:
            self.memory.save_context(
                {"input": human_input},
                {self.memory.output_key: ai_response}
            )
        except Exception as e:
            print(f"Error saving to memory: {e}")
    
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
        
        self.system_prompt = f"""You are an expert SEO assistant and web content analyzer. You answer questions based on web content that has been processed and stored in a vector database.

Context about the processed content:
- The content is sourced from live webpages (HTML parsed and chunked)
- Each chunk may include metadata such as <title>, <meta description>, <h1>, FAQ, structured data, or keyword-dense paragraphs
- The data has been embedded and stored for semantic search

Your expertise includes:
- SEO analysis: title tags, meta descriptions, headings (H1/H2), content quality, keyword usage, internal linking, and crawlability
- Content analysis: readability, user intent, content gaps, and optimization opportunities
- Technical SEO: page structure, schema markup, and HTML optimization
- General web content questions and recommendations

Instructions:
- Always provide helpful, accurate responses based on the available context
- When discussing SEO elements, mention the source URL when referring to extracted data
- If asked to improve SEO, provide actionable suggestions based on best practices
- For general questions, use your knowledge combined with the context from the processed content
- If the context doesn't provide a specific answer, clearly state what information is available and provide general guidance
- Be conversational, helpful, and avoid being overly technical unless requested
- Always aim to be constructive and solution-oriented

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
            ("human", "Based on the following context and your knowledge, please answer my question:\n\nContext: {context}\n\nQuestion: {input}")
        ])

    def setup_retrieval_chain(self):
        """Setup the retrieval chain with memory"""
        if not self.vectorizer or not self.vectorizer.vectorstore:
            raise ValueError("Vector store not available. Process URLs first.")
        
        if not self.memory_manager:
            self.memory_manager = ChatMemoryManager(vectorstore=self.vectorizer.vectorstore)
        
        retriever = self.vectorizer.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}  # Increased for better context
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

        tools = [validate_and_fetch_url, web_scraper_tool]

        # Set correct output key for tool agent
        try:
            # Create a copy of memory with correct output key for tools
            tool_memory = ConversationVectorStoreTokenBufferMemory(
                retriever=self.memory_manager.memory.retriever,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",  # Tools typically use "output"
                llm=self.memory_manager.llm,
                max_token_limit=1000,
            )
            
            self.tool_agent = initialize_agent(
                tools=tools,
                llm=self.memory_manager.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=tool_memory,
                verbose=True,
                return_direct=False,
            )
            self.use_manual_memory = False
            print("DEBUG: Agent created successfully with dedicated memory")
            
        except Exception as e:
            print(f"DEBUG: Failed to create agent with memory: {e}")
            # Fallback to agent without memory
            self.tool_agent = initialize_agent(
                tools=tools,
                llm=self.memory_manager.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                return_direct=False,
            )
            self.use_manual_memory = True
            print("DEBUG: Agent created without memory, will handle manually")

    def get_tool_response(self, user_input: str) -> Dict[str, Any]:
        """Get response using tool agent with memory"""
        try:
            if not hasattr(self, 'tool_agent'):
                self.setup_tool_agent()

            # Detect which tool might be used
            tool_used = self._detect_tool_from_input(user_input)
            
            # Only proceed with tool if a specific tool is detected
            if tool_used == 'unknown':
                return {
                    "answer": "",
                    "success": True,
                    "sources": "",
                    "tool_used": "unknown",
                    "metadata": {"tool_detection": "no_specific_tool_detected"}
                }

            # Format input for agent
            formatted_input = {"input": user_input} if isinstance(user_input, str) else user_input
                
            result = self.tool_agent.invoke(formatted_input)
            
            # Extract response text
            if isinstance(result, dict):
                response_text = (
                    result.get("output") or 
                    result.get("answer") or 
                    result.get("result") or 
                    str(result)
                )
            else:
                response_text = str(result)
            
            # Handle manual memory if needed
            if hasattr(self, 'use_manual_memory') and self.use_manual_memory:
                try:
                    self.memory_manager.add_message(user_input, response_text)
                except Exception as memory_error:
                    print(f"DEBUG: Manual memory save failed: {memory_error}")
            
            return {
                "answer": response_text,
                "success": True,
                "sources": self._extract_sources_from_response(response_text),
                "tool_used": tool_used,
                "metadata": {
                    "response_length": len(response_text),
                    "tool_detected": tool_used,
                    "memory_handling": "manual" if hasattr(self, 'use_manual_memory') and self.use_manual_memory else "automatic"
                }
            }

        except Exception as e:
            print(f"Tool agent error: {e}")
            return {
                "answer": f"Tool error: {str(e)}",
                "success": False,
                "error": str(e),
                "sources": "",
                "tool_used": "unknown",
                "metadata": {"error_type": type(e).__name__}
            }

    def _extract_sources_from_response(self, response_text: str) -> str:
        """Extract source URLs or references from response text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        urls = re.findall(url_pattern, response_text)
        
        if urls:
            return ", ".join(urls)
        
        if "✅" in response_text and "URL is valid" in response_text:
            return "URL validation tool"
        elif "🌐 Web Scraping Results" in response_text:
            return "Web scraping tool"
        
        return ""

    def _detect_tool_from_input(self, user_input: str) -> str:
        """Detect which tool might be used based on user input"""
        input_lower = user_input.lower()
        
        # More specific patterns for tool detection
        validation_keywords = ['validate url', 'check url', 'url valid', 'verify url', 'test url']
        scraping_keywords = ['scrape', 'extract content', 'get content', 'fetch content', 'scrape website']
        
        if any(keyword in input_lower for keyword in validation_keywords):
            return "validate_and_fetch_url"
        elif any(keyword in input_lower for keyword in scraping_keywords):
            return "web_scraper_tool"
        else:
            return "unknown"

    def get_response(self, user_input: str) -> Dict[str, Any]:
        """Get response with memory context using RAG"""
        try:
            if not hasattr(self, 'rag_chain'):
                self.setup_retrieval_chain()
            
            # Get chat history for context
            chat_history = []
            if self.memory_manager and hasattr(self.memory_manager.memory, 'chat_memory'):
                chat_history = self.memory_manager.memory.chat_memory.messages
            
            # Invoke the RAG chain
            response = self.rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Save to memory
            if self.memory_manager:
                self.memory_manager.add_message(user_input, response["answer"])
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("context", []),
                "success": True,
                "metadata": {
                    "context_docs_used": len(response.get("context", [])),
                    "response_type": "rag_with_memory"
                }
            }
            
        except Exception as e:
            print(f"RAG system error: {e}")
            error_message = f"I encountered an error processing your question: {str(e)}"
            
            # Still try to save error to memory
            if self.memory_manager:
                try:
                    self.memory_manager.add_message(user_input, error_message)
                except:
                    pass
                    
            return {
                "answer": error_message,
                "source_documents": [],
                "success": False,
                "error": str(e),
                "metadata": {"error_type": type(e).__name__}
            }