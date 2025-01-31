import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path so we can import from reference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reference.file_handler import process_file

async def upload_pdfs(pdf_dir: str) -> None:
    """Upload all PDFs from a directory to Pinecone."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Directory not found: {pdf_dir}")
        return

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")
        try:
            # Generate a consistent file ID based on filename and creation time
            file_id = hash(f"{pdf_file.name}_{pdf_file.stat().st_ctime}")
            created_at = datetime.fromtimestamp(pdf_file.stat().st_ctime)
            
            chunks = await process_file(
                file_path=str(pdf_file),
                file_type="application/pdf",
                file_id=file_id,
                filename=pdf_file.name,
                uploaded_by="pdf_uploader",
                message=None,
                created_at=created_at
            )
            
            if chunks:
                print(f"Successfully processed {pdf_file.name}")
                print(f"Split into {len(chunks)} chunks")
            else:
                print(f"Error: Failed to process {pdf_file.name}")
                
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_uploader.py <pdf_directory>")
        sys.exit(1)
        
    pdf_dir = sys.argv[1]
    
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)
    
    asyncio.run(upload_pdfs(pdf_dir))

if __name__ == "__main__":
    main() 