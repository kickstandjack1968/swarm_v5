import pdfplumber
import re
from typing import Optional, Dict
from pathlib import Path

def extract_invoice_data(pdf_path: str) -> Optional[Dict[str, str]]:
    """
    Extract invoice number and total amount from PDF using pdfplumber.
    
    Returns:
        Dict with 'invoice_number' and 'total_amount' or None if extraction fails
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            # Extract invoice number (looks for "Invoice #" pattern)
            invoice_match = re.search(r'Invoice\s*#\s*(\S+)', text, re.IGNORECASE)
            invoice_number = invoice_match.group(1) if invoice_match else None
            
            # Extract total amount (looks for "Total: $" pattern)
            total_match = re.search(r'Total:\s*\$([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
            total_amount = float(total_match.group(1)) if total_match else None
            
            if not invoice_number or not total_amount:
                return None
                
            return {
                "invoice_number": invoice_number,
                "total_amount": total_amount
            }
    except Exception as e:
        return None

def validate_pdf_file(pdf_path: str) -> bool:
    """Validate that the file exists and is a PDF"""
    try:
        path = Path(pdf_path)
        if not path.exists():
            return False
        if path.suffix.lower() != '.pdf':
            return False
        return True
    except Exception:
        return False
