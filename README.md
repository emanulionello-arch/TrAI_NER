# TrAI_NER

**TrAI_NER** is an interactive corpus query tool developed for an academic Natural Language Processing project. It combines **spaCy** rule- and statistical-based Named Entity Recognition (NER) pipelines with **local LLMs via Ollama** to enable semantic, natural language interrogation of annotated textual corpora.

---

## Key Features

The application is structured around two main modules:

1. **Statistical NER Dashboard:** Interactive tables and charts displaying essential corpus and entity distribution metrics across recognized categories.
2. **LLM Semantic Search & Filtering:** Sentence-level extraction filtered by NER entity types (`PERSON`, `ORG`, `LOC`/`GPE`, `MISC`), queryable through a local LLM prompt interface (e.g., *"Find all instances where [X] made statements about [Y] regarding [Z]"*).
3. **Sidebar Controls & Metrics:** Global corpus-level metrics, dataset summaries, and dynamic selector for local Ollama models.

---

## Evaluated Datasets & Corpora

The pipeline was benchmarked across three datasets:

* **Domain-Specific News Corpus (~88k tokens):** A curated dataset focusing on the early 2025 Iran–USA geopolitical conflict. It comprises English-language news articles sampled across four ideological/topical categories (*Left*, *Right*, *Finance*, and *Technology*), annotated in XML with metadata tags (`<date>`, `<author>`, `<newspaper>`, and `<title>`).
* **Synthetic Gold Standard:** A tightly controlled synthetic evaluation benchmark used for comparative zero-shot extraction testing.
* **Large-Scale Stress Corpus (~18M tokens):** A procedurally generated synthetic dataset designed to measure spaCy parsing throughput, memory usage, and batch processing latency.

---

## Utility Scripts

The repository also includes auxiliary preprocessing utilities:
* **Corpus Normalizer / Sanitizer:** Cleans non-standard typography, smart quotes, and encoding artifacts (common in texts compiled via Microsoft Word/rich text editors) to prevent XML parsing exceptions in BeautifulSoup and spaCy.
* **Synthetic Corpus Generator:** Generates high-volume text streams to test pipeline limits.

---

## Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** and [Ollama](https://ollama.com/) installed and running on your system.

### 2. Installation & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/TrAI_NER.git
cd TrAI_NER

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
