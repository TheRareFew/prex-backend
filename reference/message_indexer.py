import os
from typing import Optional
from datetime import datetime
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import logging
from app.models.message import Message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings for messages (1536 dimensions)
embeddings_1536 = OpenAIEmbeddings(model="text-embedding-ada-002")

async def index_message(message: Message) -> None:
    """Index a single message in Pinecone."""
    try:
        # Create base metadata
        metadata = {
            'message_id': str(message.id),
            'sender': message.sender.username,
            'channel': message.channel.name,
            'timestamp': message.created_at.isoformat(),
            'is_bot': message.is_bot,
            'has_attachments': message.has_attachments,
        }
        
        # Only add parent_id if it exists
        if message.parent_id:
            metadata['parent_id'] = str(message.parent_id)

        # Create document for the message
        document = Document(
            page_content=message.content,
            metadata=metadata
        )

        # Get the message index name
        message_index = os.getenv("PINECONE_INDEX_TWO")  # 1536 dimensions
        if not message_index:
            logger.error("PINECONE_INDEX_TWO environment variable not found")
            return

        # Upload to Pinecone
        Pinecone.from_documents(
            documents=[document],
            embedding=embeddings_1536,
            index_name=message_index,
            namespace="messages"
        )
        logger.info(f"Successfully indexed message {message.id} in Pinecone")

    except Exception as e:
        logger.error(f"Error indexing message {message.id} in Pinecone: {str(e)}")
        # Don't raise the exception - we don't want to break message creation if indexing fails 