import time
import os
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(LargeBinary)
    upload_datetime = Column(DateTime(timezone=True), server_default=func.now())
    category = Column(String, index=True)

    __table_args__ = (UniqueConstraint('filename', 'category', name='_filename_category_uc'),)


# Create all tables if they don't exist
Base.metadata.create_all(bind=engine)


def get_engine(retries=5, delay=2):
    for attempt in range(retries):
        try:
            engine = create_engine(DATABASE_URL)
            engine.connect()
            return engine

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise e


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
