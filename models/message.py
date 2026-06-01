from datetime import datetime
from pydantic import BaseModel

class Message(BaseModel):
    sender: str
    timestamp: datetime
    content: str
    source_file: str