from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import validators
import requests, traceback
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import mimetypes
import uuid

from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# Global variables for chat system and messages
chat_system = ""
messages = []

# Load ENV file
load_dotenv()
OPENAIKEY = os.getenv('OPEN_AI_KEY')

# Document storage configuration
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'documents')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# In-memory document store (in production, use database)
uploaded_documents = {}


system_message = """
You are an expert SEO assistant with document analysis capabilities. 
Your expertise includes:
- Keyword research and analysis
- Content optimization strategies
- Technical SEO recommendations
- Meta tag optimization
- Link building strategies
- SEO auditing and reporting
- Search ranking analysis
- Competitor analysis
- Document content analysis for SEO insights

Guidelines:
1. Only answer SEO-related questions in detail.
2. If the user asks a question unrelated to SEO, politely say you specialize in SEO and cannot answer other topics.
3. If the user clearly asks for a basic arithmetic calculation (e.g., multiplication, addition, subtraction, division), call the appropriate calculation tool.
4. If the user provides or asks to validate a URL, call the `validate_and_fetch_url` tool to check its validity and fetch its title.
5. You can analyze uploaded documents for SEO-related insights when asked.
6. Always provide actionable, practical SEO recommendations with clear steps.
"""

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    Args:
        a: first number
        b: second number
    Returns:
        The product of a and b
    """
    global chat_system
    chat_system = "Tool call - Multiply"
    return a * b

@tool
def validate_and_fetch_url(url: str) -> str:
    """Validate a URL and fetch its title if valid.
    Args:
        url: The URL to validate and fetch title from
    Returns:
        Validation result and title if successful
    """
    global chat_system
    chat_system = "Tool call - URL Validation"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
        
        return f"✅ URL is valid.\nTitle: {title}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ URL validation passed, but content fetch failed.\nError: {str(e)}"

def chatbot_view(request):
    return render(request, 'chatbot.html')

def document_view(request):
    return render(request, 'document.html')

@csrf_exempt
def upload_documents(request):
    """Handle multiple document uploads"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    if 'documents' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No documents provided'})
    
    uploaded_files = []
    errors = []
    
    # Allowed file types
    ALLOWED_EXTENSIONS = {
        'txt', 'pdf', 'doc', 'docx', 'html', 'css', 'js', 
        'json', 'xml', 'csv', 'xlsx', 'ppt', 'pptx'
    }
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    files = request.FILES.getlist('documents')
    
    for file in files:
        try:
            # Validate file size
            if file.size > MAX_FILE_SIZE:
                errors.append(f"File '{file.name}' is too large (max 10MB)")
                continue
            
            # Validate file extension
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                errors.append(f"File type '{file_extension}' not allowed for '{file.name}'")
                continue
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            file_name = f"{unique_id}_{file.name}"
            
            # Save file
            fs = FileSystemStorage(location=UPLOAD_DIR)
            filename = fs.save(file_name, file)
            file_path = fs.path(filename)
            
            # Store document metadata
            doc_info = {
                'id': unique_id,
                'original_name': file.name,
                'file_name': filename,
                'file_path': file_path,
                'size': file.size,
                'file_type': file_extension,
                'mime_type': mimetypes.guess_type(file.name)[0] or 'application/octet-stream',
                'upload_date': datetime.now().isoformat(),
                'url': fs.url(filename) if hasattr(fs, 'url') else None
            }
            
            uploaded_documents[unique_id] = doc_info
            uploaded_files.append({
                'id': unique_id,
                'name': file.name,
                'size': file.size,
                'type': file_extension,
                'upload_date': doc_info['upload_date']
            })
            
        except Exception as e:
            errors.append(f"Error uploading '{file.name}': {str(e)}")
    
    return JsonResponse({
        'success': len(uploaded_files) > 0,
        'uploaded_files': uploaded_files,
        'errors': errors,
        'total_uploaded': len(uploaded_files),
        'total_errors': len(errors)
    })

@csrf_exempt
def list_documents(request):
    """Return list of uploaded documents"""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests allowed'})
    
    documents_list = []
    for doc_id, doc_info in uploaded_documents.items():
        documents_list.append({
            'id': doc_id,
            'name': doc_info['original_name'],
            'size': doc_info['size'],
            'type': doc_info['file_type'],
            'upload_date': doc_info['upload_date'],
            'mime_type': doc_info['mime_type']
        })
    
    # Sort by upload date (newest first)
    documents_list.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return JsonResponse({
        'success': True,
        'documents': documents_list,
        'total_count': len(documents_list)
    })

@csrf_exempt
def delete_document(request):
    """Delete an uploaded document"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    document_id = request.POST.get('document_id', '').strip()
    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'})
    
    if document_id not in uploaded_documents:
        return JsonResponse({'success': False, 'error': 'Document not found'})
    
    try:
        doc_info = uploaded_documents[document_id]
        
        # Delete physical file
        if os.path.exists(doc_info['file_path']):
            os.remove(doc_info['file_path'])
        
        # Remove from memory store
        del uploaded_documents[document_id]
        
        return JsonResponse({
            'success': True,
            'message': f"Document '{doc_info['original_name']}' deleted successfully"
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f"Error deleting document: {str(e)}"
        })

@csrf_exempt
def chatbot_input(request):
    """SEO-focused chatbot with arithmetic tool support"""
    global chat_system

    if not hasattr(request, 'POST') or request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

    usermessage = request.POST.get('message', '').strip()
    if not usermessage:
        return JsonResponse({'success': False, 'error': 'No message provided'})

    try:
        # SEO + arithmetic tool
        tools = [multiply, validate_and_fetch_url]

        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=OPENAIKEY,
            model="gpt-3.5-turbo"
        )
        llm_with_tools = llm.bind_tools(tools)
        chat_system = "LLM Call"
                
        # Initial messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=usermessage)
        ]

        # Get initial AI response
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        print(f"AI Response: {ai_msg}")
        print(f"Tool calls: {ai_msg.tool_calls}")

        # Process tool calls if any
        if ai_msg.tool_calls:
            tool_mapping = {
                "multiply": multiply,
                "validate_and_fetch_url": validate_and_fetch_url
            }

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                print(f"Processing tool call: {tool_name} with args: {tool_args}")
                
                if tool_name in tool_mapping:
                    selected_tool = tool_mapping[tool_name]
                    try:
                        # Run the tool with args
                        tool_result = selected_tool.invoke(tool_args)
                        print(f"Tool result: {tool_result}")
                        
                        # Append ToolMessage with matching tool_call_id
                        messages.append(
                            ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                        )
                    except Exception as e:
                        print(f"Error running tool {tool_name}: {str(e)}")
                        messages.append(
                            ToolMessage(content=f"❌ Error running tool {tool_name}: {str(e)}", tool_call_id=tool_call_id)
                        )
                else:
                    messages.append(
                        ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call_id)
                    )
            
            # Get final response after tool execution
            final_response = llm_with_tools.invoke(messages)
            print(f"Final response: {final_response}")
        else:
            # No tools called, use initial response
            final_response = ai_msg

        return JsonResponse({
            'success': True,
            'response': final_response.content,
            'session_id': "12345",
            'message_id': "12345",
            'response_meta': {
                'source': chat_system,
                'response_type': 'Tool Usage' if ai_msg.tool_calls else 'General Conversation',
                'has_vectorstore': False,
                'tools_used': len(ai_msg.tool_calls) > 0 if ai_msg.tool_calls else False,
                'tool_calls_made': [tc["name"] for tc in ai_msg.tool_calls] if ai_msg.tool_calls else [],
                'total_tokens': final_response.usage_metadata.get("total_tokens", 0),
                'input_tokens': final_response.usage_metadata.get("input_tokens", 0),
                'output_tokens': final_response.usage_metadata.get("output_tokens", 0)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })