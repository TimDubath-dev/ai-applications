# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZHAW AI Applications course - Week 3: ML model deployment with Gradio. Two projects that train scikit-learn models in Jupyter notebooks and serve predictions via Gradio web UIs.

## Architecture

### Iris Classifier (`iris/`)
- **`iris.ipynb`** - Trains a `RandomForestClassifier` on the Iris dataset, saves model as pickle
- **`app.py`** - Standalone Gradio app that loads `iris_random_forest_classifier.pkl` and serves predictions
- Input: 4 numeric features (sepal/petal length/width) → Output: species name

### Apartment Price Predictor (`apartment/`)
- **`apartment.ipynb`** - Loads a pre-trained `RandomForestRegressor` from pickle, joins with BFS municipality data (`bfs_municipality_and_tax_data.csv`), and serves a Gradio UI
- Input: rooms, area, town (dropdown of Zürich-area municipalities) → Output: predicted price
- Features used: rooms, area, pop, pop_dens, frg_pct, emp, tax_income
- No separate `app.py`; the Gradio app runs directly from the notebook

## Commands

```bash
# Run the iris Gradio app
cd iris && python app.py

# Install iris dependencies
pip install -r iris/requirements.txt  # only lists scikit-learn; also needs gradio, pandas

# Run notebooks
jupyter notebook iris/iris.ipynb
jupyter notebook apartment/apartment.ipynb
```

## Key Details

- Models are persisted as `.pkl` files via Python's `pickle` module
- The apartment model expects a specific sklearn version (trained with 1.6.1); version mismatches produce warnings
- The apartment notebook relies on `bfs_municipality_and_tax_data.csv` being in the same directory, with a `tax_income` column that needs string-to-float conversion (removing apostrophes)
- Gradio launches on localhost (default ports 7860/7861)
