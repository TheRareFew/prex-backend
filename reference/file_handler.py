import os
from typing import Optional, List, Tuple
import aiofiles
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from PIL import Image
import io
import fitz  # PyMuPDF for PDF processing
import base64
from openai import AsyncOpenAI
from datetime import datetime
from app.models.message import Message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LLM for text processing
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Initialize embeddings
embeddings_1536 = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings_3072 = OpenAIEmbeddings(model="text-embedding-3-large")

async def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return content
    except UnicodeDecodeError:
        # If UTF-8 fails, try with a different encoding
        async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
            content = await f.read()
            return content

async def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file."""
    try:
        text = []
        doc = fitz.open(file_path)
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        raise


async def generate_text_summary(content: str) -> str:
    """Generate a single summary for the entire text content."""
    try:
        # Limit content length for the summary
        max_content_length = 4000  # Adjust based on model's context window
        truncated_content = content[:max_content_length] + ("..." if len(content) > max_content_length else "")
        
        prompt = f"Please provide a concise summary of the following text (max 200 words):\n\n{truncated_content}"
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating text summary: {str(e)}")
        return "Error generating summary"


async def upload_to_pinecone_with_model(
    documents: List[Document],
    embeddings,
    index_name: str,
    namespace: str
) -> None:
    """Upload documents to Pinecone with specified embedding model."""
    try:
        Pinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )
        logger.info(f"Successfully uploaded {len(documents)} documents to Pinecone index {index_name}")
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {str(e)}")
        pass

async def process_file(
    file_path: str,
    file_type: str,
    file_id: int,
    filename: str,
    uploaded_by: str,
    message: Optional[Message],
    created_at: datetime
) -> Optional[Tuple[List[str], List[str]]]:
    """Process a file and generate its description."""
    try:
        logger.info(f"Starting file processing for {filename} (type: {file_type})")
        description = None
        raw_chunks = None
        
        # Process based on file type
        if file_type.startswith('text/') or file_type == 'application/pdf':
            logger.info(f"Processing text/PDF file: {filename}")
            content = await read_pdf_file(file_path) if file_type == 'application/pdf' else await read_text_file(file_path)
            
            # Get raw chunks for 3072d index
            raw_chunks = await process_text_content(content)
            # Generate one summary for the entire document
            description = await generate_text_summary(content)
            
            # Create documents for raw chunks (3072d)
            raw_documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'file_id': str(file_id),
                        'filename': filename,
                        'uploaded_by': uploaded_by,
                        'file_type': file_type,
                        'upload_date': str(created_at),
                        'chunk_index': i,
                        'total_chunks': len(raw_chunks),
                        'content_type': 'raw_chunk',
                        **({"message_text": message.content} if message else {})
                    }
                )
                for i, chunk in enumerate(raw_chunks)
            ]
            
            # Create single document for description (3072d)
            description_document = Document(
                page_content=description,
                metadata={
                    'file_id': str(file_id),
                    'filename': filename,
                    'uploaded_by': uploaded_by,
                    'file_type': file_type,
                    'upload_date': str(created_at),
                    'content_type': 'description',
                    **({"message_text": message.content} if message else {})
                }
            )
            
            # Upload raw chunks and description to 3072d index in separate namespaces
            pinecone_index_3072 = os.getenv("PINECONE_INDEX")
            
            # Upload chunks to chunks namespace
            await upload_to_pinecone_with_model(
                raw_documents,
                embeddings_3072,
                pinecone_index_3072,
                "chunks"
            )
            
            # Upload description to descriptions namespace
            await upload_to_pinecone_with_model(
                [description_document],
                embeddings_3072,
                pinecone_index_3072,
                "descriptions"
            )
            
            
        else:
            logger.warning(f"Unsupported file type for processing: {file_type}")
            return None
        
        if not description:
            logger.warning(f"No description generated for file {filename}")
            
        return [description] if description else None, raw_chunks
    
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
        return None 