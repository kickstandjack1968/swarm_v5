from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import os
from datetime import datetime

from src.database import engine, get_db
from src.models import Invoice
from src.schemas import InvoiceCreate, InvoiceRead
from src.utils import extract_invoice_data, validate_pdf_file

app = FastAPI(title="PDF Invoice Processor", version="1.0.0")

# Create tables
Invoice.__table__.create(bind=engine, checkfirst=True)

@app.post("/extract", response_model=InvoiceRead)
async def extract_invoice(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Extract invoice data from PDF and store in database.
    
    Returns:
        Invoice data including ID, filename, invoice number, total amount, and creation timestamp
    """
    
    # Validate file type
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    temp_filename = f"temp_{datetime.now().timestamp()}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate PDF
        if not validate_pdf_file(temp_path):
            raise HTTPException(status_code=400, detail="Invalid PDF file")
            
        # Extract data from PDF
        extracted_data = extract_invoice_data(temp_path)
        
        if not extracted_data:
            raise HTTPException(status_code=400, detail="Could not extract invoice data from PDF")
        
        # Create and save invoice record
        db_invoice = Invoice(
            filename=file.filename,
            invoice_number=extracted_data["invoice_number"],
            total_amount=extracted_data["total_amount"]
        )
        
        db.add(db_invoice)
        db.commit()
        db.refresh(db_invoice)
        
        return db_invoice
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/invoices", response_model=List[InvoiceRead])
async def get_invoices(db: Session = Depends(get_db)):
    """Get all stored invoices"""
    return db.query(Invoice).all()

@app.get("/invoices/{invoice_id}", response_model=InvoiceRead)
async def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    """Get a specific invoice by ID"""
    db_invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not db_invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return db_invoice

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
