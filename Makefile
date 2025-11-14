# Makefile for Information Retrieval Search Engine
# Project: French Wikipedia Search Engine with TF-IDF

PYTHON := python
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
DATA_DIR := wiki_split_extract_2k
QUERIES_FILE := requetes.jsonl

.PHONY: help setup install download-model test search compare clean

help:
	@echo "================================================================"
	@echo "Information Retrieval Search Engine - Makefile"
	@echo "================================================================"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make setup           - Complete setup (venv + dependencies + model)"
	@echo "  make search          - Interactive search demo"
	@echo "  make compare         - Compare different configurations"
	@echo "  make clean           - Remove generated files and cache"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup        - Setup environment (first time only)"
	@echo "  2. make search       - Interactive search demo"
	@echo "  3. make compare      - Compare configurations"
	@echo ""
	@echo "================================================================"

setup: $(VENV) install download-model
	@echo ""
	@echo "================================================================"
	@echo "Setup complete!"
	@echo "================================================================"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make test' to evaluate the search engine"
	@echo "  - Run 'make search' for interactive search"
	@echo "  - Run 'make compare' to compare configurations"
	@echo ""

$(VENV):
	@echo "Creating Python virtual environment..."
	python3.12 -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)/"

install: $(VENV)
	@echo ""
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install spacy scikit-learn matplotlib numpy
	@echo "Dependencies installed successfully"

download-model: $(VENV)
	@echo ""
	@echo "Downloading spaCy French language model..."
	$(PYTHON_VENV) -m spacy download fr_core_news_sm
	@echo "Language model downloaded successfully"

search: $(VENV)
	@echo ""
	@echo "================================================================"
	@echo "Starting interactive search engine..."
	@echo "================================================================"
	@echo ""
	$(PYTHON_VENV) search_engine.py
	@echo ""
	@echo "================================================================"
	@echo "Search session ended"
	@echo "================================================================"

compare: $(VENV)
	@echo ""
	@echo "================================================================"
	@echo "Comparing different feature configurations..."
	@echo "================================================================"
	@echo ""
	$(PYTHON_VENV) compare_features.py
	@echo ""
	@echo "================================================================"
	@echo "Comparison complete - Check generated PNG files"
	@echo "================================================================"

clean:
	@echo "Cleaning generated files..."
	rm -f *.png
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Complete cleanup done"

# Tutorial commands for step-by-step setup
tutorial-step1:
	@echo "================================================================"
	@echo "STEP 1: Environment Setup"
	@echo "================================================================"
	@echo ""
	@echo "This will create a Python virtual environment with Python 3.12"
	@echo "and install all required dependencies."
	@echo ""
	@echo "Run: make setup"
	@echo ""
	@echo "This installs:"
	@echo "  - spaCy (NLP library for French text processing)"
	@echo "  - scikit-learn (TF-IDF baseline for comparison)"
	@echo "  - matplotlib (visualization)"
	@echo "  - numpy (numerical operations)"
	@echo "  - fr_core_news_sm (French language model)"
	@echo ""

tutorial-step2:
	@echo "================================================================"
	@echo "STEP 2: Test the Search Engine"
	@echo "================================================================"
	@echo ""
	@echo "Once setup is complete, test the search engine with:"
	@echo ""
	@echo "  make test"
	@echo ""
	@echo "This will:"
	@echo "  - Load 2000 Wikipedia documents from $(DATA_DIR)/"
	@echo "  - Build TF-IDF index with text cleaning and lemmatization"
	@echo "  - Evaluate on queries from $(QUERIES_FILE)"
	@echo "  - Display Precision@10, Recall@10, and F1@10 metrics"
	@echo ""

tutorial-step3:
	@echo "================================================================"
	@echo "STEP 3: Interactive Search"
	@echo "================================================================"
	@echo ""
	@echo "Try the search engine interactively:"
	@echo ""
	@echo "  make search"
	@echo ""
	@echo "This demonstrates the search engine with example queries."
	@echo "The engine uses:"
	@echo "  - TF-IDF vectorization with log(1+tf) normalization"
	@echo "  - Inverted index for efficient term lookup"
	@echo "  - Cosine similarity for document ranking"
	@echo "  - Text preprocessing (stopword removal, lemmatization)"
	@echo ""

tutorial-step4:
	@echo "================================================================"
	@echo "STEP 4: Compare Configurations"
	@echo "================================================================"
	@echo ""
	@echo "Compare different feature combinations:"
	@echo ""
	@echo "  make compare"
	@echo ""
	@echo "This compares 6 configurations:"
	@echo "  1. Manual TF-IDF (baseline)"
	@echo "  2. Manual TF-IDF with log normalization"
	@echo "  3. Scikit-learn TF-IDF (baseline)"
	@echo "  4. Manual with text cleaning"
	@echo "  5. Manual with text cleaning + log"
	@echo "  6. Scikit-learn with text cleaning"
	@echo ""
	@echo "Generates 3 graphs: precision, recall, and F1 scores"
	@echo ""

tutorial-full: tutorial-step1 tutorial-step2 tutorial-step3 tutorial-step4
	@echo "================================================================"
	@echo "Complete Tutorial"
	@echo "================================================================"
	@echo ""
	@echo "To get started, run these commands in order:"
	@echo ""
	@echo "  1. make setup      - Install dependencies"
	@echo "  2. make search     - Try interactive search"
	@echo "  3. make compare    - Compare configurations"
	@echo ""
	@echo "For help at any time: make help"
	@echo ""
	@echo "================================================================"
