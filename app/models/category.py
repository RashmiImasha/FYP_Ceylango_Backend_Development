from pydantic import BaseModel
from typing import Optional

class Category(BaseModel):
    category_name: str
    category_type: str
    category_image: Optional[str]

class CategoryOut(Category):
    id: str
