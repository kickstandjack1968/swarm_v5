from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from src.database import Base

class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    invoice_number = Column(String)
    total_amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
