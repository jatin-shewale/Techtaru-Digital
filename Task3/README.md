# Data Extraction Bot

Convert resume or invoice text into structured JSON, validate the extracted fields, and review warnings for missing or inconsistent data. Task3 includes two ways to use the same extraction pipeline:

- **Streamlit UI** for interactive text entry and results.
- **FastAPI service** for applications that need an HTTP endpoint.

Groq generates the structured response. Pydantic validates it against the selected schema, and project-specific checks flag likely data-quality issues.

## What it extracts

### Resume

- Name, email, and phone
- Skills
- Education entries with institution, degree, and year
- Experience entries with company, title, dates, and summary

### Invoice

- Invoice number, vendor, customer, issue date, due date, and currency
- Line items with description, quantity, unit price, and amount
- Subtotal, tax, and total

Fields absent from the source are left `null` or returned as empty lists. The extractor is instructed not to invent missing values.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access to reach Groq

## Setup on Windows PowerShell

Run from the repository root:

```powershell
cd Task3
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `Task3/.env` and set your Groq key:

```dotenv
GROQ_API_KEY=gsk-your-key-here
```

The default chat model is set in `.env.example`. Keep the API key private; `.env` is ignored by Git.

## Run the Streamlit application

From the `Task3` directory, with the virtual environment active:

```powershell
python -m streamlit run streamlit_app.py
```

Choose **Resume** or **Invoice**, paste text or upload a PDF or UTF-8 `.txt` file, then select **Extract fields**. PDFs are read page by page, with page markers kept in the text sent for extraction. The results page shows key fields, any validation warnings, the complete JSON, and a JSON download button. The Streamlit app calls `extraction_core.py` directly; FastAPI does not need to be running.

Scanned PDFs must be OCRed before upload. Password-protected PDFs must be unlocked first. Image uploads are not supported.

## Run the FastAPI service

From the `Task3` directory:

```powershell
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The service also has a health check at `GET /health`.

### Extract an invoice

PowerShell example:

```powershell
$body = @{ text = 'INVOICE #42 Acme Ltd to Beta Corp. Widget x2 at $5 = $10. Subtotal $10 Tax $1 Total $11 USD' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/extract/invoice -ContentType 'application/json' -Body $body
```

Use `POST /extract/resume` for a resume. Both endpoints accept a JSON object with a `text` property and return:

```json
{
  "data": {
    "invoice_number": "42",
    "vendor": "Acme Ltd",
    "customer": "Beta Corp",
    "issue_date": null,
    "due_date": null,
    "currency": "USD",
    "line_items": [
      {
        "description": "Widget",
        "quantity": 2,
        "unit_price": 5,
        "amount": 10
      }
    ],
    "subtotal": 10,
    "tax": 1,
    "total": 11
  },
  "warnings": []
}
```

The exact fields depend on the document type and source text.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Required | Authenticates calls to the Groq API. |
| `CHAT_MODEL` | `openai/gpt-oss-20b` | Groq-hosted chat model used for extraction. Choose a Groq model that supports JSON Schema output. |

The `openai/` prefix is part of this Groq model identifier; the application uses the Groq SDK and Groq API.

## How the pipeline works

1. **Choose the schema.** `extraction_core.py` maps the requested document type to a Pydantic `Resume` or `Invoice` model.
2. **Bound the input.** Empty text and unknown document types are rejected. Text longer than 20,000 characters is truncated to that limit to reduce the chance of exceeding Groq's token-throughput limits. The UI shows a notice when it will truncate the input.
3. **Ask Groq for structured JSON.** The Pydantic JSON Schema is sent with the document text using Groq's JSON Schema response format. The default model is configured for best-effort schema output, then Pydantic parses and validates the returned JSON. Invalid or empty model output becomes an extraction error.
4. **Check business rules.** The app checks resume email format and missing names, invoice line-item arithmetic, line-item totals against subtotal, and subtotal plus tax against total. These checks produce warnings; they do not silently rewrite model output.
5. **Return or render the result.** The API returns `data` and `warnings`. The Streamlit interface renders common fields, warnings, full JSON, and a downloadable JSON file.

Groq receives the document text (up to 20,000 characters after truncation) to perform extraction. Do not submit confidential information unless your organisation permits sending it to the configured API provider.

## Error handling

| Condition | API behaviour | Streamlit behaviour |
|---|---|---|
| Empty text or unknown document type | HTTP 422 | Shows an input error |
| Groq rate limit | Retries up to five attempts; then HTTP 429 | Retries in the extraction layer, then shows a rate-limit message |
| Groq connection or timeout failure | Retries up to five attempts; then HTTP 502 | Retries, then shows a connection message |
| Request exceeds Groq's token limit | HTTP 413 with a shorter-input message | Prompts you to shorten or split the document |
| Refusal, empty response, or invalid JSON/schema | HTTP 502 | Shows the extraction error |
| Missing API key | Configuration error | Shows the setup message |

## Project files

| File | Purpose |
|---|---|
| `extraction_core.py` | Pydantic schemas, Groq request, output parsing, and validation rules. |
| `api.py` | FastAPI service with health and extraction endpoints. |
| `streamlit_app.py` | Interactive Streamlit application. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Environment variable template. |

## Troubleshooting

**`ModuleNotFoundError: No module named 'groq'`**: activate the Task3 virtual environment, then run `python -m pip install -r requirements.txt` from the `Task3` directory.

**The `api` module cannot be imported**: change into the `Task3` directory before starting Uvicorn. The command must run beside `api.py`.

**Authentication or model errors**: check that `Task3/.env` contains a valid `GROQ_API_KEY` and that `CHAT_MODEL` names a model available to your Groq account with JSON Schema output support.

**A PDF has no extractable text**: scanned PDFs need OCR first. Password-protected PDFs must be unlocked before upload.

**Fields are empty or warnings appear**: review the source text and the returned JSON. The model can only extract details that appear in the supplied text, and validation warnings flag inconsistencies rather than correcting them.
