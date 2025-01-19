from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

# Database connection URL
DATABASE_URL = "postgresql://postgres:postgres@localhost:5435/langroid"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for models
Base = declarative_base()

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Example: Create a simple model
class ExampleModel(Base):
    __tablename__ = 'example'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    embed = Column(Vector)

# Create tables (if they don't exist)
Base.metadata.create_all(bind=engine)

# Create a session and interact with the database
with SessionLocal() as session:
    # Add multiple entries
    entries = [
        ExampleModel(name="Alice",embed=[1,2,3]),
        ExampleModel(name="Bob",embed=[1,2,3]),
        ExampleModel(name="Charlie",embed=[1,2,3])
    ]
    
    session.add_all(entries)
    session.commit()

    # Query data
    results = session.query(ExampleModel).all()
    for result in results:
        print(f"ID: {result.id}, Name: {result.name}")

