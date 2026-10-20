"""Extract and validate resume or invoice fields using Groq and Pydantic."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Type

from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


load_dotenv()
EXTRACT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
# Keep prompt size conservative for Groq's on-demand token-throughput limits.
MAX_INPUT_CHARS = 20_000


# Rate limits and transient network errors are retried with exponential backoff.
api_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    # max_retries=0: tenacity (above) owns the retry policy so behaviour is explicit.
    return Groq(api_key=key, timeout=30, max_retries=0)


class ExtractionError(Exception):
    """The model refused, or returned nothing usable."""


# ------------------------------------------------------------------ schemas
class Education(BaseModel):
    institution: str
    degree: Optional[str] = Field(None, description="e.g. B.Tech Computer Science")
    year: Optional[int] = Field(None, description="Graduation year")


class Experience(BaseModel):
    company: str
    title: str
    start: Optional[str] = Field(None, description="YYYY-MM if known")
    end: Optional[str] = Field(None, description="YYYY-MM, or null if current")
    summary: Optional[str] = None


class Resume(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    vendor: Optional[str] = None
    customer: Optional[str] = None
    issue_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD")
    currency: Optional[str] = Field(None, description="ISO code, e.g. USD, INR")
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None


SCHEMAS: Dict[str, Type[BaseModel]] = {"resume": Resume, "invoice": Invoice}

SYSTEM = (
    "Extract the requested fields from the text. Use ONLY information present in the text. "
    "If a field is missing, use null (or an empty list). Never guess or invent values. "
    "Normalise dates to ISO format and numbers to plain numerics."
)


# --------------------------------------------------------------- extraction
def _inline_schema(schema: dict) -> dict:
    """Inline Pydantic $defs so Groq receives a self-contained JSON Schema."""
    definitions = schema.get("$defs", {})

    def expand(value):
        if isinstance(value, dict):
            if "$ref" in value:
                ref_name = value["$ref"].rsplit("/", 1)[-1]
                return expand(definitions[ref_name])
            return {key: expand(item) for key, item in value.items() if key != "$defs"}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(schema)


@api_retry
def _call(client: Groq, text: str, schema: Type[BaseModel]):
    return client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__.lower(),
                "strict": False,
                "schema": _inline_schema(schema.model_json_schema()),
            },
        },
        temperature=0,
    )


def extract(text: str, doc_type: str, client: Optional[Groq] = None) -> BaseModel:
    """Extract structured data. Raises ValueError for bad input, ExtractionError for model failure."""
    if doc_type not in SCHEMAS:
        raise ValueError(f"Unknown doc_type '{doc_type}'. Choose from {list(SCHEMAS)}.")
    text = (text or "").strip()
    if not text:
        raise ValueError("Input text is empty.")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]  # keep cost/latency bounded

    completion = _call(client or get_client(), text, SCHEMAS[doc_type])
    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise ExtractionError(f"Model refused: {msg.refusal}")
    if not msg.content:
        raise ExtractionError("Model returned no parsable output.")
    try:
        return SCHEMAS[doc_type].model_validate_json(msg.content)
    except (ValueError, TypeError) as e:
        raise ExtractionError("Model returned JSON that did not match the expected schema.") from e


# ---------------------------------------------------------- sanity checks
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate(result: BaseModel) -> List[str]:
    """Business-rule checks the schema cannot express. Returns a list of warnings."""
    warnings: List[str] = []
    if isinstance(result, Resume):
        if result.email and not _EMAIL.match(result.email):
            warnings.append(f"Email looks invalid: {result.email}")
        if not result.full_name:
            warnings.append("No name found.")
    if isinstance(result, Invoice):
        items_sum = round(sum(i.amount for i in result.line_items), 2)
        if result.subtotal is not None and result.line_items and abs(items_sum - result.subtotal) > 0.01:
            warnings.append(f"Line items sum to {items_sum} but subtotal is {result.subtotal}.")
        if None not in (result.subtotal, result.tax, result.total):
            if abs(result.subtotal + result.tax - result.total) > 0.01:
                warnings.append("subtotal + tax does not equal total.")
        for i in result.line_items:
            if abs(i.quantity * i.unit_price - i.amount) > 0.01:
                warnings.append(f"'{i.description}': qty x price != amount.")
    return warnings
