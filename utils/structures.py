"""Helper library for structured output definitions."""
from pydantic import BaseModel

class Extraction(BaseModel):
    """Represents an structured answer for the model output"""

    url: str
    pii_information_present: bool
    information: str
