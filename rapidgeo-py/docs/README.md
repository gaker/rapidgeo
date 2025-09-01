# rapidgeo Documentation

This directory contains the Sphinx documentation for rapidgeo.

## Building Documentation Locally

1. Install documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the rapidgeo package first (required for autodoc):
   ```bash
   cd ..
   maturin develop --features pyo3/extension-module
   ```

3. Build the documentation:
   ```bash
   make html
   ```

4. View the documentation:
   ```bash
   open _build/html/index.html
   ```

## Read the Docs

Documentation is automatically built and hosted at https://rapidgeo.readthedocs.io/ when changes are pushed to the main branch.

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `distance.rst` - Distance calculation documentation
- `polyline.rst` - Polyline encoding documentation  
- `simplify.rst` - Line simplification documentation
- `performance.rst` - Performance guide
- `api.rst` - Complete API reference