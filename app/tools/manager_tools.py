from typing import Dict, Any, Optional, Union
from langchain.tools import tool
from pydantic import UUID4
from datetime import datetime
import logging
import re
from app.core.database import get_supabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up LLM for article generation
article_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def make_slug(text: str) -> str:
    """Create URL-friendly slug from text."""
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces with hyphens
    slug = re.sub(r'[\s]+', '-', slug)
    # Remove special characters
    slug = re.sub(r'[^\w\-]', '', slug)
    # Remove multiple hyphens
    slug = re.sub(r'\-+', '-', slug)
    # Trim hyphens from ends
    slug = slug.strip('-')
    return slug

@tool
async def write_article(prompt: str) -> Dict[str, Any]:
    """Generate an article based on the manager's prompt. Article content should be formatted in HTML for rich text. Do not use markdown. Article title and description should not have any HTML tags."""
    try:
        # First, generate title and description
        title_desc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional knowledge base article writer. 
            Generate a clear, concise title and description for an article based on the given prompt.
            The title should be under 100 characters and capture the main point.
            The description should be 1-2 sentences summarizing the key points.
            Format: 
            Title: [title]
            Description: [description]"""),
            ("human", "{prompt}")
        ])
        
        title_desc_result = await article_llm.ainvoke(
            title_desc_prompt.format_messages(prompt=prompt)
        )
        title_desc_text = title_desc_result.content
        
        # Parse title and description
        title_match = re.search(r'Title: (.+)', title_desc_text)
        desc_match = re.search(r'Description: (.+)', title_desc_text)
        
        title = title_match.group(1) if title_match else prompt[:97] + "..."
        description = desc_match.group(1) if desc_match else prompt[:197] + "..."

        # Generate article content
        content_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional knowledge base article writer.
            Write a well-structured article based on the given prompt.
            Format the content in clean HTML with proper semantic structure:
            - Use <h1> for the main title
            - Use <h2> for major sections
            - Use <h3> for subsections if needed
            - Use <p> for paragraphs
            - Use <ul>/<li> for lists
            - Add appropriate CSS classes for styling (kb-article or faq-article)
            - Make content clear, professional, and well-organized
            - Include practical examples or steps if relevant
            - Format code snippets with <pre><code> if any
            
            The article should be comprehensive but concise."""),
            ("human", """Title: {title}
            Description: {description}
            Prompt: {prompt}""")
        ])
        
        content_result = await article_llm.ainvoke(
            content_prompt.format_messages(
                title=title,
                description=description,
                prompt=prompt
            )
        )
        content = content_result.content

        # Determine if FAQ based on content and prompt
        is_faq = any([
            prompt.lower().startswith(('how', 'what', 'why', 'when', 'where', 'can')),
            '?' in prompt,
            'FAQ' in content,
            'Frequently Asked' in content
        ])

        # Determine category based on content
        category_keywords = {
            'product': ['product', 'feature', 'release', 'version', 'upgrade'],
            'service': ['service', 'support', 'help', 'assistance'],
            'troubleshooting': ['issue', 'problem', 'error', 'fix', 'solve', 'debug'],
            'faq': ['faq', 'question', 'ask', 'common'],
            'policy': ['policy', 'rule', 'guideline', 'requirement', 'compliance']
        }
        
        category = 'general'
        content_lower = content.lower()
        for cat, keywords in category_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                category = cat
                break
        if is_faq:
            category = 'faq'

        # Generate slug from title
        slug = make_slug(title)

        # Current timestamp
        current_time = datetime.utcnow().isoformat()

        return {
            "action": "write_article",
            "article": {
                "title": title,
                "description": description,
                "content": content,
                "status": "pending_approval",
                "created_at": current_time,
                "updated_at": current_time,
                "published_at": None,
                "view_count": 0,
                "is_faq": is_faq,
                "category": category,
                "slug": slug
            }
        }
    except Exception as e:
        logger.error(f"Article generation error: {str(e)}")
        raise

@tool
async def update_article_status(article_id: UUID4, new_status: str, reason: str = "") -> Dict[str, Any]:
    """Update the status of an article."""
    try:
        valid_statuses = ["draft", "pending_approval", "approved", "rejected", "archived"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

        current_time = datetime.utcnow().isoformat()
        
        # If article is being approved, set published_at
        published_at = current_time if new_status == "approved" else None

        return {
            "action": "update_article_status",
            "article_id": str(article_id),
            "status": new_status,
            "reason": reason,
            "metadata": {
                "updated_at": current_time,
                "published_at": published_at
            }
        }
    except Exception as e:
        logger.error(f"Article status update error: {str(e)}")
        raise

@tool
async def add_article_note(article_id: UUID4, note: str) -> Dict[str, Any]:
    """Add a note to an article."""
    try:
        return {
            "action": "add_article_note",
            "article_id": str(article_id),
            "note": note,
            "metadata": {
                "created_at": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Add article note error: {str(e)}")
        raise

def get_fields_for_table(table_name: str, requested_fields: Optional[str] = None) -> str:
    """Get appropriate fields for table based on context and requested fields."""
    # Define default field sets for each table
    default_fields = {
        "articles": {
            "minimal": "id,title,status,created_at",
            "standard": "id,title,description,status,created_at,updated_at,category,slug",
            "full": "id,title,description,content,status,created_at,updated_at,published_at,view_count,is_faq,category,slug,created_by"
        },
        "article_notes": {
            "minimal": "id,article_id,created_at",
            "standard": "id,article_id,content,created_at",
            "full": "id,article_id,content,created_at,created_by"
        },
        "article_versions": {
            "minimal": "id,article_id,version_number",
            "standard": "id,article_id,title,version_number,created_at",
            "full": "id,article_id,title,description,content,created_at,created_by,version_number,change_summary"
        },
        "approval_requests": {
            "minimal": "id,article_id,status",
            "standard": "id,article_id,status,submitted_at,reviewed_at",
            "full": "id,article_id,version_id,submitted_by,submitted_at,reviewed_by,reviewed_at,status,feedback"
        },
        "article_tags": {
            "minimal": "id,article_id,tag",
            "standard": "id,article_id,tag",
            "full": "id,article_id,tag"
        },
        "manager_prompts": {
            "minimal": "id,conversation_id",
            "standard": "id,conversation_id,prompt,created_at",
            "full": "id,conversation_id,prompt,created_at"
        },
        "manager_responses": {
            "minimal": "id,prompt_id",
            "standard": "id,prompt_id,response,created_at",
            "full": "id,prompt_id,response,created_at"
        },
        "response_notes": {
            "minimal": "id,response_id",
            "standard": "id,response_id,note,created_at",
            "full": "id,response_id,note,created_at,created_by"
        }
    }

    # If specific fields requested, use those
    if requested_fields and requested_fields != "*":
        return requested_fields

    # Get default fields for table
    table_fields = default_fields.get(table_name, {})
    
    # If table not found, return all fields
    if not table_fields:
        return "*"

    # Default to standard fields unless specific needs detected
    field_set = "standard"

    # Use full fields if table is articles and we're likely doing content search
    if table_name == "articles":
        field_set = "full"  # Always get full article data to avoid multiple queries

    return table_fields[field_set]

async def query_table(
    table_name: str,
    fields: str = "*",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Base function to query tables with filters."""
    try:
        # Get appropriate fields
        fields = get_fields_for_table(table_name, fields)
        
        # Log incoming parameters
        logger.info(f"query_table called with: table={table_name}, fields={fields}, filters={filters}")
        
        sb = get_supabase()
        query = sb.table(table_name).select(fields).order(order_by, desc=descending)
        
        # Convert string filters to dict if needed
        if isinstance(filters, str):
            if "=" in filters and not filters.startswith("{"):
                key, value = filters.split("=", 1)
                # Remove quotes if present
                value = value.strip("'\"")
                filters = {key: value}
            else:
                try:
                    import json
                    filters = json.loads(filters)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse filters string: {filters}")
                    filters = None
        
        # Build filters carefully with better error handling
        if filters:
            try:
                for col, condition in filters.items():
                    logger.info(f"Applying filter: {col}={condition}")
                    
                    # Special handling for title searches
                    if col == "title" and table_name == "articles":
                        # Use ilike for case-insensitive partial matches
                        query = query.ilike(col, f"%{condition}%")
                        continue
                    
                    if "__" in col:  # Support operators like eq, gt, lt, etc
                        col, op = col.split("__")
                        query = query.filter(col, op, condition)
                    else:
                        query = query.eq(col, condition)
            except Exception as filter_error:
                logger.error(f"Filter error: {str(filter_error)}, filters={filters}")
                raise ValueError(f"Invalid filter format: {str(filter_error)}")
        
        # Add pagination
        query = query.range(offset, offset + limit - 1)
        result = query.execute()
        
        return {
            "action": f"query_{table_name}",
            "table": table_name,
            "data": result.data,
            "metadata": {
                "count": len(result.data),
                "total_estimated": result.count if hasattr(result, 'count') else None,
                "queried_at": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Table query error on {table_name}: {str(e)}, params: table={table_name}, fields={fields}, filters={filters}")
        raise

@tool
async def query_articles(
    fields: Optional[str] = None,
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query articles table. Schema:
    - id (uuid)
    - title (text)
    - description (text, nullable)
    - content (text)
    - status (draft, pending_approval, approved, rejected, archived)
    - created_by (uuid)
    - created_at (timestamptz)
    - updated_at (timestamptz)
    - published_at (timestamptz, nullable)
    - view_count (integer)
    - is_faq (boolean)
    - category (general, product, service, troubleshooting, faq, policy, other)
    - slug (text)

    Example usage:
    - Get pending articles: filters={"status": "pending_approval"}
    - Get recent articles: filters={"view_count__gt": 100}
    - Get specific article: filters={"id": "some-uuid"}
    - Get article by title: filters={"title__ilike": "%search term%"}

    Do not include content in fields unless asked for.
    """
    return await query_table("articles", fields, filters, order_by, descending, limit, offset)

@tool
async def query_article_notes(
    fields: Optional[str] = None,
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query article notes table. Schema:
    - id (uuid)
    - article_id (uuid)
    - content (text)
    - created_at (timestamptz)
    - created_by (uuid)

    Example usage:
    - Get notes for article: filters={"article_id": "some-uuid"}
    - Get recent notes: filters={"created_at__gt": "2024-01-01"}
    - Get notes by creator: filters={"created_by": "user-uuid"}
    """
    return await query_table("article_notes", fields, filters, order_by, descending, limit, offset)

@tool
async def query_article_versions(
    fields: str = "id,article_id,title,description,content,created_at,created_by,version_number,change_summary",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query article versions table. Schema:
    - id (uuid)
    - article_id (uuid)
    - title (text)
    - description (text, nullable)
    - content (text)
    - created_at (timestamptz)
    - created_by (uuid)
    - version_number (integer)
    - change_summary (text, nullable)

    Example usage:
    - Get versions for article: filters={"article_id": "some-uuid"}
    - Get latest versions: filters={"version_number__gt": 1}
    - Get versions by author: filters={"created_by": "user-uuid"}
    """
    return await query_table("article_versions", fields, filters, order_by, descending, limit, offset)

@tool
async def query_approval_requests(
    fields: str = "id,article_id,version_id,submitted_by,submitted_at,reviewed_by,reviewed_at,status,feedback",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "submitted_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query approval requests table. Schema:
    - id (uuid)
    - article_id (uuid)
    - version_id (uuid)
    - submitted_by (uuid)
    - submitted_at (timestamptz)
    - reviewed_by (uuid, nullable)
    - reviewed_at (timestamptz, nullable)
    - status (pending, approved, rejected)
    - feedback (text, nullable)

    Example usage:
    - Get pending requests: filters={"status": "pending"}
    - Get requests for article: filters={"article_id": "some-uuid"}
    - Get requests by reviewer: filters={"reviewed_by": "user-uuid"}
    """
    return await query_table("approval_requests", fields, filters, order_by, descending, limit, offset)

@tool
async def query_article_tags(
    fields: str = "id,article_id,tag",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query article tags table. Schema:
    - id (uuid)
    - article_id (uuid)
    - tag (text)

    Example usage:
    - Get tags for article: filters={"article_id": "some-uuid"}
    - Get articles with tag: filters={"tag": "some-tag"}
    """
    return await query_table("article_tags", fields, filters, order_by, descending, limit, offset)

@tool
async def query_manager_prompts(
    fields: str = "id,conversation_id,prompt,created_at",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query manager prompts table. Schema:
    - id (uuid)
    - conversation_id (uuid)
    - prompt (text)
    - created_at (timestamptz)

    Example usage:
    - Get prompts for conversation: filters={"conversation_id": "some-uuid"}
    - Get recent prompts: filters={"created_at__gt": "2024-01-01"}
    """
    return await query_table("manager_prompts", fields, filters, order_by, descending, limit, offset)

@tool
async def query_manager_responses(
    fields: str = "id,prompt_id,response,created_at",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query manager responses table. Schema:
    - id (uuid)
    - prompt_id (uuid)
    - response (text)
    - created_at (timestamptz)

    Example usage:
    - Get responses for prompt: filters={"prompt_id": "some-uuid"}
    - Get recent responses: filters={"created_at__gt": "2024-01-01"}
    """
    return await query_table("manager_responses", fields, filters, order_by, descending, limit, offset)

@tool
async def query_response_notes(
    fields: str = "id,response_id,note,created_at,created_by",
    filters: Optional[Union[Dict[str, Any], str]] = None,
    order_by: str = "created_at",
    descending: bool = True,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """Query response notes table. Schema:
    - id (uuid)
    - response_id (uuid)
    - created_by (uuid)
    - note (text)
    - created_at (timestamptz)

    Example usage:
    - Get notes for response: filters={"response_id": "some-uuid"}
    - Get notes by creator: filters={"created_by": "user-uuid"}
    """
    return await query_table("response_notes", fields, filters, order_by, descending, limit, offset) 