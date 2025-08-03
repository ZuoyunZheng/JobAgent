# JobAgent

JobAgent is an intelligent automation tool designed to streamline the job application process by combining RAG (Retrieval-Augmented Generation), job scraping, and browser automation technologies.

## Features

The system consists of three main components:

### 1. RAG Pipeline (`rag.py`)
- **Purpose**: Extracts and retrieves candidate information from PDF documents using vector embeddings
- **Technology**: LangChain + PostgreSQL with pgvector + Google Gemini embeddings
- **Functionality**:
  - Loads candidate CV/resume from PDF format
  - Chunks documents using RecursiveCharacterTextSplitter (500 chars, 100 overlap)
  - Stores embeddings in PostgreSQL vector database using Google Gemini embeddings
  - Provides semantic search capabilities for candidate information
- **Status**: ✅ Tested and Working

### 2. Job Scraping (`scrape.py`)
- **Purpose**: Scrapes job listings and filters them based on candidate profile relevance
- **Technology**: python-jobspy + LangChain + Google Gemini LLM
- **Functionality**:
  - Scrapes jobs from LinkedIn and Glassdoor
  - Analyzes job descriptions using Google Gemini LLM
  - Retrieves relevant candidate qualifications via RAG
  - Scores job-candidate fit ("yes", "maybe", "no")
  - Filters and returns relevant opportunities
- **Status**: ✅ Tested and Working

### 3. Browser Automation (`browser.py`)
- **Purpose**: Automatically applies to jobs by filling application forms
- **Technology**: browser-use + Google Chrome + Google Gemini LLM
- **Functionality**:
  - Opens job application pages
  - Reads candidate CV for context
  - Fills application forms automatically
  - Asks user for missing information
  - Defers to human approval before submission
- **Status**: ✅ Framework Ready (requires GUI environment for testing)

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Google AI API key for Gemini models
- Google Chrome browser
- Environment variables in `.env`:
  ```
  POSTGRES_USER=your_username
  POSTGRES_PASSWORD=your_password
  GOOGLE_API_KEY=your_google_ai_api_key  # Required for Gemini LLM and embeddings
  ```

## Setup

1. **Install Dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

2. **Setup Database**:
   ```bash
   # Ensure PostgreSQL is running
   sudo systemctl start postgresql

   # Install pgvector extension (if not already installed)
   sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Setup Environment Variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file with your actual values
   nano .env  # or use your preferred editor
   ```
   - Get your Google AI API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
   - Fill in your PostgreSQL credentials
   - Optionally configure other settings as needed

4. **Prepare Candidate Data**:
   - Place candidate CV/resume as PDF in `data/` directory
   - Update the PDF path in `rag.py` (currently set to `data/ZuoyunZhengCVplain.pdf`)

## Usage

### Quick Start - Run All Components
```bash
source .venv/bin/activate  # or activate your virtual environment
python main.py --reload_data
```

### Individual Components

#### 1. Initialize RAG Pipeline Only
```bash
python main.py --skip_scraping --skip_browser --reload_data
```
This will:
- Create/recreate the PostgreSQL database
- Load and chunk the candidate's CV
- Generate and store embeddings
- Test the retrieval system with sample queries

#### 2. Scrape and Filter Jobs Only
```bash
python main.py --skip_rag --skip_browser --job_terms "Data Scientist" --location "Berlin" --num_jobs 10
```
This will:
- Scrape ML/Data Science jobs from LinkedIn
- Analyze each job description
- Compare against candidate profile using RAG
- Filter and display relevant opportunities
- Cache results in `jobs.pkl`

#### 3. Browser Automation Only (GUI Required)
```bash
python main.py --skip_rag --skip_scraping
```
This will:
- Open a browser instance
- Navigate to job application pages
- Fill forms using candidate information
- Request user input for missing data
- Wait for approval before submission

### Command Line Options
```bash
python main.py --help
```
Available options:
- `--reload_data`: Reload and re-index candidate data
- `--skip_rag`: Skip RAG pipeline initialization
- `--skip_scraping`: Skip job scraping
- `--skip_browser`: Skip browser automation
- `--job_terms`: Job search terms (default: "Machine Learning")
- `--location`: Job search location (default: "Germany")
- `--num_jobs`: Number of jobs to scrape (default: 20)
- `--save_jobs`: Save scraped jobs to jobs.pkl file

## Project Structure

```
JobAgent/
├── rag.py              # RAG pipeline for candidate information
├── scrape.py           # Job scraping and filtering
├── browser.py          # Browser automation for applications
├── main.py             # Main entry point for all components
├── utils/
│   └── args.py         # Command line argument parsing
├── scripts/
│   ├── bu.py           # Browser debugging utilities
│   ├── pw.py           # Playwright testing scripts
│   └── langhchain_pw.py # LangChain + Playwright integration
├── data/               # Candidate documents (symlinked)
├── jobs.pkl            # Cached job listings
└── pyproject.toml      # Project dependencies
```

## Configuration

Key configuration parameters can be modified in the respective files:

- **RAG settings** (`rag.py`): chunk_size (500), chunk_overlap (100), similarity search (k=3), text-embedding-004 model
- **Job scraping** (`scrape.py`): search terms ("Machine Learning"), location ("Germany"), results (20), gemini-2.5-flash-lite model
- **Browser automation** (`browser.py`): Chrome path, LinkedIn URLs, gemini-2.5-flash-lite model

## Previous Iterations

### LLMSherpa (Deprecated)
LLMSherpa was a previous solution for PDF document chunking and processing in the RAG pipeline. It provided advanced document parsing capabilities through the nlm-ingestor service:

```bash
# Previous LLMSherpa setup (no longer used)
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
```

The current implementation has moved to LangChain's PyPDFLoader for better integration and simplified deployment, while maintaining the same core functionality for document processing and embedding generation.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests and ensure code quality with `ruff`
4. Submit a pull request

## License

This project is open source. Please check the license file for details.
