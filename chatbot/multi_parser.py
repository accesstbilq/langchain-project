import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from urllib.parse import urljoin
import time
import concurrent.futures
from threading import Lock
import warnings

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders.sitemap import SitemapLoader

# Suppress the relevance score warning
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

# Replace with your actual API key

@dataclass
class WebPageMetadata:
    """Data class to store webpage metadata"""
    title: str
    description: str
    keywords: str
    url: str
    content_length: int
    headings: Dict[str, List[str]]
    faqs: List[Dict[str, str]]
    processing_status: str = "success"
    error_message: str = ""

class WebContentParser:
    """Parse web content, extract metadata, and handle embeddings"""
    
    def __init__(self, url: str, timeout: int = 30, embedding_model: str = "openai", 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.url = url
        self.timeout = timeout
        self.soup = None
        self.metadata = None
        
        # Embedding configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-3-small")
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
        
    def fetch_content(self) -> bool:
        """Fetch webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
            return True
        except Exception as e:
            print(f"Error fetching content from {self.url}: {e}")
            return False
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the webpage"""
        if not self.soup:
            return {}
        
        # Extract title
        title = self.soup.find('title')
        title_text = title.text.strip() if title else ""
        
        # Extract meta description
        meta_desc = self.soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ""
        
        # Extract meta keywords
        meta_keywords = self.soup.find('meta', attrs={'name': 'keywords'})
        keywords = meta_keywords.get('content', '') if meta_keywords else ""
        
        # Extract Open Graph data
        og_data = {}
        og_tags = self.soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            og_data[property_name] = tag.get('content', '')
        
        return {
            'title': title_text,
            'description': description,
            'keywords': keywords,
            'url': self.url,
            'og_data': og_data
        }
    
    def extract_headings(self) -> Dict[str, List[str]]:
        """Extract all heading tags (H1-H6) from the webpage"""
        headings = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
        
        if not self.soup:
            return headings
        
        for level in range(1, 7):
            heading_tags = self.soup.find_all(f'h{level}')
            headings[f'h{level}'] = [tag.get_text().strip() for tag in heading_tags if tag.get_text().strip()]
        
        return headings
    
    def extract_faqs(self) -> List[Dict[str, str]]:
        """Extract FAQ content from the webpage"""
        faqs = []
        
        if not self.soup:
            return faqs
        
        # Common FAQ patterns
        faq_patterns = [
            {'class': re.compile(r'faq', re.I)},
            {'id': re.compile(r'faq', re.I)},
            {'class': re.compile(r'question', re.I)},
            {'class': re.compile(r'accordion', re.I)},
        ]
        
        for pattern in faq_patterns:
            faq_sections = self.soup.find_all(attrs=pattern)
            for section in faq_sections:
                questions = section.find_all(['h3', 'h4', 'h5', 'strong', 'b'])
                for q in questions:
                    question_text = q.get_text().strip()
                    if len(question_text) > 10 and '?' in question_text:
                        answer_element = q.find_next_sibling(['p', 'div'])
                        if not answer_element:
                            answer_element = q.parent.find_next_sibling(['p', 'div'])
                        
                        answer_text = answer_element.get_text().strip() if answer_element else ""
                        
                        if answer_text and len(answer_text) > 20:
                            faqs.append({
                                'question': question_text,
                                'answer': answer_text
                            })
        
        # Remove duplicates
        seen = set()
        unique_faqs = []
        for faq in faqs:
            faq_key = faq['question'].lower()
            if faq_key not in seen:
                seen.add(faq_key)
                unique_faqs.append(faq)
        
        return unique_faqs
    
    def extract_main_content(self) -> str:
        """Extract main content from the webpage"""
        if not self.soup:
            return ""
        
        # Remove script and style elements
        for script in self.soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get main content areas
        main_content = ""
        content_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '#content',
            '.post-content',
            '.entry-content'
        ]
        
        for selector in content_selectors:
            content_area = self.soup.select_one(selector)
            if content_area:
                main_content = content_area.get_text()
                break
        
        # If no specific content area found, get body text
        if not main_content:
            body = self.soup.find('body')
            main_content = body.get_text() if body else self.soup.get_text()
        
        # Clean up the text
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        return main_content
    
    def create_documents_from_content(self, content: str) -> List[Document]:
        """Create LangChain documents from parsed content"""
        documents = []
        
        if not content.strip():
            return documents
        
        # Split main content into chunks
        content_chunks = self.text_splitter.split_text(content)
        
        # Create documents from chunks
        for i, chunk in enumerate(content_chunks):
            chunk_len = len(chunk.strip())
            
            if chunk_len >= 100:  # Minimum chunk size
                metadata = {
                    'source': self.url,
                    'title': self.metadata.title if self.metadata else "",
                    'chunk_index': i,
                    'content_type': 'main_content',
                    'chunk_size': chunk_len
                }
                documents.append(Document(page_content=chunk.strip(), metadata=metadata))
        
        # Add heading information as separate documents
        if self.metadata:
            all_headings_text = ""
            for level, headings in self.metadata.headings.items():
                for heading in headings:
                    if len(heading.strip()) >= 10:
                        all_headings_text += f"{level.upper()}: {heading.strip()}\n"
            
            if all_headings_text and len(all_headings_text) >= 50:
                metadata = {
                    'source': self.url,
                    'title': self.metadata.title,
                    'content_type': 'headings',
                    'chunk_size': len(all_headings_text)
                }
                documents.append(Document(page_content=all_headings_text, metadata=metadata))
            
            # Add FAQ content as separate documents
            all_faqs_text = ""
            for faq in self.metadata.faqs:
                faq_content = f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
                all_faqs_text += faq_content
            
            if all_faqs_text and len(all_faqs_text) >= 50:
                metadata = {
                    'source': self.url,
                    'title': self.metadata.title,
                    'content_type': 'faqs',
                    'chunk_size': len(all_faqs_text)
                }
                documents.append(Document(page_content=all_faqs_text.strip(), metadata=metadata))
        
        return documents
    
    def parse(self) -> WebPageMetadata:
        """Parse the webpage and extract all information"""
        try:
            if not self.fetch_content():
                return WebPageMetadata(
                    title="", description="", keywords="", url=self.url,
                    content_length=0, headings={'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []},
                    faqs=[], processing_status="failed", error_message="Failed to fetch content"
                )
            
            metadata = self.extract_metadata()
            headings = self.extract_headings()
            faqs = self.extract_faqs()
            content = self.extract_main_content()
            
            self.metadata = WebPageMetadata(
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                keywords=metadata.get('keywords', ''),
                url=self.url,
                content_length=len(content),
                headings=headings,
                faqs=faqs,
                processing_status="success"
            )
            
            return self.metadata
            
        except Exception as e:
            return WebPageMetadata(
                title="", description="", keywords="", url=self.url,
                content_length=0, headings={'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []},
                faqs=[], processing_status="failed", error_message=str(e)
            )

class MultiURLVectorizer:
    """Handle multiple URLs and create a unified vector store"""
    
    def __init__(self, urls: List[str], embedding_model: str = "openai", chunk_size: int = 1000, 
                 max_workers: int = 5, delay_between_requests: float = 1.0):
        self.urls = urls
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests
        self.vectorstore = None
        self.processing_results = {}
        self.lock = Lock()
        
    def process_single_url(self, url: str) -> Dict[str, Any]:
        """Process a single URL and return results"""
        try:
            print(f"Processing URL: {url}")
            
            # Add delay to be respectful to servers
            time.sleep(self.delay_between_requests)
            
            parser = WebContentParser(url, embedding_model=self.embedding_model, chunk_size=self.chunk_size)
            metadata = parser.parse()
            
            if metadata.processing_status == "failed":
                return {
                    'url': url,
                    'status': 'failed',
                    'error': metadata.error_message,
                    'documents': [],
                    'metadata': metadata
                }
            
            # Extract main content and create documents
            content = parser.extract_main_content()
            documents = parser.create_documents_from_content(content)
            
            return {
                'url': url,
                'status': 'success',
                'documents': documents,
                'metadata': metadata,
                'document_count': len(documents),
                'parser': parser  # Keep reference for embeddings
            }
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return {
                'url': url,
                'status': 'failed',
                'error': str(e),
                'documents': [],
                'metadata': None
            }
    
    def process_urls_sequential(self) -> Dict[str, Any]:
        """Process URLs sequentially"""
        all_documents = []
        successful_urls = []
        failed_urls = []
        embeddings_obj = None
        
        for url in self.urls:
            result = self.process_single_url(url)
            self.processing_results[url] = result
            
            if result['status'] == 'success':
                all_documents.extend(result['documents'])
                successful_urls.append(url)
                if not embeddings_obj and 'parser' in result:
                    embeddings_obj = result['parser'].embeddings
                print(f"✓ Successfully processed {url} - {result['document_count']} documents created")
            else:
                failed_urls.append(url)
                print(f"✗ Failed to process {url}: {result.get('error', 'Unknown error')}")
        
        return {
            'all_documents': all_documents,
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'total_documents': len(all_documents),
            'embeddings': embeddings_obj
        }
    
    def process_urls_parallel(self) -> Dict[str, Any]:
        """Process URLs in parallel using ThreadPoolExecutor"""
        all_documents = []
        successful_urls = []
        failed_urls = []
        embeddings_obj = None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all URL processing tasks
            future_to_url = {executor.submit(self.process_single_url, url): url for url in self.urls}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    self.processing_results[url] = result
                    
                    if result['status'] == 'success':
                        with self.lock:
                            all_documents.extend(result['documents'])
                            successful_urls.append(url)
                            if not embeddings_obj and 'parser' in result:
                                embeddings_obj = result['parser'].embeddings
                        print(f"✓ Successfully processed {url} - {result['document_count']} documents created")
                    else:
                        failed_urls.append(url)
                        print(f"✗ Failed to process {url}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_urls.append(url)
                    print(f"✗ Exception processing {url}: {e}")
        
        return {
            'all_documents': all_documents,
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'total_documents': len(all_documents),
            'embeddings': embeddings_obj
        }
    
    def process(self, parallel: bool = True) -> Dict[str, Any]:
        """Process all URLs and create a unified vector store"""
        print(f"Processing {len(self.urls)} URLs...")
        print(f"Processing mode: {'Parallel' if parallel else 'Sequential'}")
        
        # Process URLs
        if parallel:
            processing_result = self.process_urls_parallel()
        else:
            processing_result = self.process_urls_sequential()
        
        all_documents = processing_result['all_documents']
        successful_urls = processing_result['successful_urls']
        failed_urls = processing_result['failed_urls']
        embeddings_obj = processing_result['embeddings']
        
        print(f"\nProcessing Summary:")
        print(f"  Successful URLs: {len(successful_urls)}")
        print(f"  Failed URLs: {len(failed_urls)}")
        print(f"  Total Documents: {len(all_documents)}")
        
        # Create unified vector store if we have documents
        if all_documents and embeddings_obj:
            print(f"\nCreating unified vector store with {len(all_documents)} documents...")
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=all_documents,
                    embedding=embeddings_obj,
                    collection_name="multi_url_web_content",
                    persist_directory="./chroma_db_multi"
                )

                collection = self.vectorstore.get()
                print(f"✓ Vector store created successfully!")
                print(f"Collection name: {self.vectorstore._collection.name}")
                print(f"Number of documents: {len(collection['ids'])}")
                print("="*60)
            except Exception as e:
                print(f"✗ Error creating vector store: {e}")
                raise
        else:
            print("⚠ No documents to create vector store from!")
        
        # Compile summary statistics
        summary_stats = {
            'total_urls_processed': len(self.urls),
            'successful_urls': len(successful_urls),
            'failed_urls': len(failed_urls),
            'total_documents_created': len(all_documents),
            'successful_url_list': successful_urls,
            'failed_url_list': failed_urls,
            'vector_store_location': './chroma_db_multi' if all_documents else None,
            'processing_details': {}
        }
        
        # Add detailed stats for each URL
        for url, result in self.processing_results.items():
            if result['status'] == 'success' and result['metadata']:
                metadata = result['metadata']
                summary_stats['processing_details'][url] = {
                    'title': metadata.title,
                    'content_length': metadata.content_length,
                    'document_count': result['document_count'],
                    'headings_count': sum(len(headings) for headings in metadata.headings.values()),
                    'faqs_count': len(metadata.faqs),
                    'status': 'success'
                }
            else:
                summary_stats['processing_details'][url] = {
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                }
        
        return summary_stats
    
    # def test_vector_store(self, query: str = "test", k: int = 1):
    #     """Test the vector store with different retrieval methods"""
    #     if not self.vectorstore:
    #         raise Exception("Vector store not created. Run process() first.")
        
    #     print(f"\n{'='*60}")
    #     print("TESTING VECTOR STORE RETRIEVAL")
    #     print(f"{'='*60}")
    #     print(f"Query: '{query}'")
    #     print(f"{'='*60}")
        
    #     # Max Marginal Relevance (MMR) search
    #     print("Method 1: MMR (Maximum marginal relevance retrieval) search for diverse results")
    #     try:
    #         retriever = self.vectorstore.as_retriever(
    #             search_type="mmr",
    #             search_kwargs={"k": k}
    #         )
    #         docs = retriever.invoke(query)
    #         print(f"Found {len(docs)} documents via MMR")
    #         for i, doc in enumerate(docs):
    #             print(f"  Document {i+1}:")
    #             print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    #             print(f"    Content Type: {doc.metadata.get('content_type', 'Unknown')}")
    #             print(f"    Preview: {doc.page_content}...")
    #             print()
    #     except Exception as e:
    #         print(f"Error in MMR search: {e}")
        
    #     print(f"{'='*60}")
    
    def search_similar(self, query: str, k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content in the vector store using basic similarity search"""
        if not self.vectorstore:
            raise Exception("Vector store not created. Run process() first.")
        
        try:
            if source_filter:
                # Use basic similarity search and filter results manually
                all_results = self.vectorstore.similarity_search(query, k=k*2)  # Get more to filter
                results = [doc for doc in all_results if doc.metadata.get('source') == source_filter][:k]
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source_url': doc.metadata.get('source', 'Unknown')
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in search_similar: {e}")
            return []
    
    def get_documents_by_source(self, source_url: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source URL"""
        if not self.vectorstore:
            raise Exception("Vector store not created. Run process() first.")
        
        try:
            # Use similarity search with a broad query to get all documents
            # then filter by source
            all_docs = self.vectorstore.similarity_search("", k=1000)  # Large k to get all
            
            source_docs = []
            for doc in all_docs:
                if doc.metadata.get('source') == source_url:
                    source_docs.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    })
            
            return source_docs
            
        except Exception as e:
            print(f"Error in get_documents_by_source: {e}")
            return []

# Example usage and demo
# def main():
#     """Main function to demonstrate multi-URL processing"""

#     urls = ["https://www.brihaspatitech.com/ecommerce-development-company/"]
#     print("="*60)

#     try:
#         # Initialize the multi-URL vectorizer
#         vectorizer = MultiURLVectorizer(
#             urls=urls, 
#             embedding_model="openai",
#             chunk_size=1000,
#             delay_between_requests=1.0  # 1 second delay between requests
#         )
        
#         # Process all URLs (parallel processing)
#         vectorizer.process(parallel=True)

#         # Test different queries
#         test_queries = [
#             "Le Bouquet",
#             "How long does it take to build an ecommerce store?"
#         ]
        
#         for query in test_queries:
#             print(f"\n🔍 Testing query: '{query}'")
#             vectorizer.test_vector_store(query, k=1)
            
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()