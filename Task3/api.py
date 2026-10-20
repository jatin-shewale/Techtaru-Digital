"""api.py - FastAPI wrapper around the extraction bot.  Run:  uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel

from extraction_core import ExtractionError, extract, validate

app = FastAPI(title="Data Extraction Bot", version="1.0")


class ExtractRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract/{doc_type}")
def extract_endpoint(doc_type: str, body: ExtractRequest):
    try:
        result = extract(body.text, doc_type)
    except ValueError as e:                       # empty text / unknown type
        raise HTTPException(status_code=422, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Upstream rate limit. Retry shortly.")
    except (APIConnectionError, APITimeoutError):
        raise HTTPException(status_code=502, detail="Upstream Groq API unavailable. Retry shortly.")
    except APIStatusError as e:
        if e.status_code == 413:
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the current Groq token limit. Send a shorter text section.",
            )
        raise HTTPException(status_code=502, detail="Upstream Groq API returned an error.")
    except ExtractionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"data": result.model_dump(), "warnings": validate(result)}
