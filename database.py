from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Define the database path
DATABASE_FILE = "recommender.db"
# SQLite URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# Create the SQLAlchemy engine
# 'check_same_thread=False' is required for SQLite with FastAPI/multithreading
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative class definitions
Base = declarative_base()

# --- Database Models ---

class ProductDB(Base):
    """Maps to the 'product_catalog' CSV data."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True) # Internal primary key
    product_id = Column(Integer, unique=True, nullable=False) # The ID from the CSV
    name = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String, nullable=True) # Optional field

class BehaviorDB(Base):
    """Maps to the 'user_behavior_log' CSV data."""
    __tablename__ = "behaviors"

    id = Column(Integer, primary_key=True, index=True) # Internal primary key
    user_id = Column(String, index=True, nullable=False)
    action_type = Column(String, nullable=False)
    value = Column(String, nullable=False)

# --- Initialization Function ---

def init_db():
    """Create all tables in the database."""
    # This creates the .db file if it doesn't exist and defines the tables
    Base.metadata.create_all(bind=engine)

# Dependency to get a database session
def get_db() -> Generator[Session, None, None]:
    """Provides a database session for a request and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Run the initialization when the server starts (or when this file is imported)
init_db()