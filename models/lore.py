from pydantic import BaseModel, Field
from typing import Optional

class Quote(BaseModel):
    text: str
    speaker: str
    source_file: str

class Location(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class Character(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    is_pc: bool = False
    player_name: Optional[str] = None
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class HistoryEvent(BaseModel):
    description: str
    scope: str
    chronological_position: Optional[int] = None
    supporting_quotes: list[Quote] = Field(default_factory=list)

class OtherDetail(BaseModel):
    detail: str
    supporting_quotes: list[Quote] = Field(default_factory=list)