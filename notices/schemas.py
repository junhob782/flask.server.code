#스키마
from dataclasses import dataclass
from typing import Optional

@dataclass
class NoticeCreate:
    title: str
    content: str
    is_pinned: bool = False

@dataclass
class NoticeUpdate:
    title: Optional[str] = None
    content: Optional[str] = None
    is_pinned: Optional[bool] = None