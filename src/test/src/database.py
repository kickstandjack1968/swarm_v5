# src/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///calibration.db')
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()
