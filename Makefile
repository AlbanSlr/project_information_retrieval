# Makefile for Information Retrieval Search Engine
# Project: French Wikipedia Search Engine with TF-IDF (Optimized)

PYTHON := python
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
DATA_DIR := wiki_split_extract_2k
QUERIES_FILE := requetes.jsonl
CACHE_DIR := data

.PHONY: help setup install download-models search compare-features compare-models compare clean clean-cache

help:
	@echo "================================================================"
	@echo "Information Retrieval Search Engine - Makefile"
	@echo "================================================================"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make setup           - Complete setup (venv + dependencies + models)"
	@echo "  make search          - Interactive search with TF-IDF (log + cleaning)"
	@echo "  make compare-features - Compare TF-IDF feature configurations"
	@echo "  make compare-models  - Compare different models (TF-IDF, SBERT, Hybrid)"
	@echo "  make compare         - Run both comparisons"
	@echo "  make clean           - Remove generated files"
	@echo "  make clean-cache     - Remove cached models and vectors"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup            - Setup environment (first time only)"
	@echo "  2. make search           - Try the optimized TF-IDF search"
	@echo "  3. make compare-features - Compare TF-IDF configurations"
	@echo "  4. make compare-models   - Compare TF-IDF vs SBERT vs Hybrid"
	@echo ""
	@echo "================================================================"

setup: $(VENV) install download-models
	@echo ""
	@echo "================================================================"
	@echo "Setup complete!"
	@echo "================================================================"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make search' for interactive TF-IDF search"
	@echo "  - Run 'make compare-features' to compare TF-IDF configurations"
	@echo "  - Run 'make compare-models' to compare different models"
	@echo ""
	@echo "Note: Main engine is TF-IDF (log + cleaning) - optimal performance"
	@echo ""

$(VENV):
	@echo "Creating Python virtual environment..."
	python3.12 -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)/"

install: $(VENV)
	@echo ""
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install spacy scikit-learn matplotlib numpy sentence-transformers
	@echo "Dependencies installed successfully"

download-models: $(VENV)
download-models: $(VENV)
	@echo ""
	@echo "Downloading required models..."
	$(PYTHON_VENV) -m spacy download fr_core_news_sm
	@echo "Models downloaded successfully"
	@echo ""

search: $(VENV)
	@echo ""
	@echo "================================================================"
	@echo "Starting TF-IDF search engine (log + cleaning)"
	@echo "================================================================"
	@echo ""
	$(PYTHON_VENV) search_engine.py
	@echo ""
	@echo "================================================================"
	@echo "Search session ended"
	@echo "================================================================"

compare-features: $(VENV)
	@echo ""
	@echo "================================================================"
	@echo "Comparing TF-IDF feature configurations..."
	@echo "================================================================"
	@echo ""
	$(PYTHON_VENV) compare_features.py
	@echo ""
	@echo "================================================================"
	@echo "Feature comparison complete - Check tfidf_features_comparison.png"
	@echo "================================================================"

compare-models: $(VENV)
	@echo ""
	@echo "================================================================"
	@echo "Comparing search models (TF-IDF, SBERT, Hybrid)..."
	@echo "================================================================"
	@echo ""
	$(PYTHON_VENV) compare_models.py
	@echo ""
	@echo "================================================================"
	@echo "Model comparison complete - Check model_comparison.png"
	@echo "================================================================"

compare: compare-features compare-models
	@echo ""
	@echo "================================================================"
	@echo "All comparisons complete!"
	@echo "Generated files:"
	@echo "  - tfidf_features_comparison.png (TF-IDF feature ablation)"
	@echo "  - model_comparison.png (Model comparison)"
	@echo "================================================================"

clean:
	@echo "Cleaning generated files..."
	rm -f *.png
	rm -rf __pycache__
	rm -rf engines/__pycache__
	rm -rf utils/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"

clean-cache: clean
	@echo "Removing cached models and vectors..."
	rm -rf $(CACHE_DIR)
	@echo "Cache cleaned - models will be retrained on next run"

clean-all: clean clean-cache
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Complete cleanup done"