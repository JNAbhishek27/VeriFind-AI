# VeriFind AI - Intelligent Web Information Retrieval Agent

**VeriFind AI** is an intelligent retrieval agent built for the **AMD AI League Match 4: Intelligent Web Information Retrieval**. It accepts natural-language questions, navigates websites, extracts exact factual information, and delivers evidence-backed answers with verifiable source URLs and confidence scoring.

---

## 1. Project Architecture

```
verifind-ai/
├── app.py                     # Streamlit Frontend (Branding, Controls, Citations)
├── requirements.txt           # Production & Development dependencies
├── .env.example               # Template for environment secrets & configurations
├── .gitignore                 # Standard Python, IDE, OS, & Playwright ignore rules
├── README.md                  # System documentation & usage guide
│
├── backend/
│   ├── __init__.py            # Core dataclasses (RetrievalResult, EvidenceItem, VerificationStatus)
│   ├── query_parser.py        # Natural-language intent normalization & query breakdown
│   ├── web_search.py          # Search dispatcher & candidate URL discovery engine
│   ├── browser.py             # Playwright headless browser automation & HTTP fallback
│   ├── extractor.py           # Content extraction via Trafilatura & BeautifulSoup4
│   ├── ranker.py              # Passage relevance scoring & heuristic ranking
│   ├── answer_engine.py       # Orchestrator coordinating the end-to-end retrieval pipeline
│   └── verifier.py            # Cross-source factual evidence verification & consensus checks
│
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py      # Unit tests for HTML cleaning & paragraph extraction
│   └── test_verifier.py       # Unit tests for evidence verification and consensus scoring
│
└── assets/                    # Static assets, branding logos, and diagrams
```

---

## 2. Technical Stack

- **Python Runtime:** Python 3.14 (Compatible with Python 3.11+)
- **Frontend:** Streamlit 1.67+
- **Browser Automation:** Playwright Chromium with HTTPX fallback
- **Content Extraction:** Trafilatura & BeautifulSoup4 (lxml)
- **Data Validation & Typing:** Pydantic v2 & Python Dataclasses
- **Testing Framework:** Pytest
- **Future AI Expansion:** Google Gemini API & AMD ROCm-accelerated local inference backend

---

## 3. Installation & Setup

### Prerequisites
- macOS (Apple Silicon M2 supported)
- Homebrew Python 3.11+

### 1. Clone or Open the Workspace
```bash
cd /Users/abhishekjn/Desktop/verifind-ai
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: Install Playwright Browsers
If you wish to use dynamic JavaScript rendering via Playwright:
```bash
playwright install chromium
```
*(Note: If Playwright binaries are omitted, VeriFind AI automatically falls back to an asynchronous HTTPX fetcher to ensure continuous operation.)*

### 5. Configure Environment Variables
```bash
cp .env.example .env
```
Populate any API keys (e.g. `GEMINI_API_KEY`) when ready for subsequent AI reasoning phases.

---

## 4. Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 5. Running the Test Suite

Run unit tests via `pytest`:
```bash
pytest tests -v
```

All unit tests currently validate:
- HTML title and text body extraction via Trafilatura / BeautifulSoup.
- Noise filtering (removal of nav, footer, scripts).
- Evidence verification scoring, cross-source corroboration, and confidence levels.

