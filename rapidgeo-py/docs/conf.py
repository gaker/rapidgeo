# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# For Read the Docs: Don't add local path, use installed package
# For local development: Add path if package not installed
if 'READTHEDOCS' not in os.environ:
    try:
        import rapidgeo
    except ImportError:
        sys.path.insert(0, os.path.abspath('../python'))

# Project information
project = 'rapidgeo'
copyright = f'{datetime.now().year}, Greg Aker'
author = 'Greg Aker'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

try:
    import rapidgeo
    version = rapidgeo.__version__
    release = rapidgeo.__version__
except ImportError:
    version = 'unknown'
    release = 'unknown'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Suppress specific warnings
suppress_warnings = [
    # 'ref.python',  # Suppress cross-reference warnings for Python objects
    # 'toc.not_included',  # Suppress toctree not included warnings
]

# Autodoc type hints configuration
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary configuration
autosummary_generate = True

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# # Theme options
# html_theme_options = {
#     'canonical_url': '',
#     'analytics_id': '',
#     'logo_only': False,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     'vcs_pageview_mode': '',
#     'style_nav_header_background': 'white',
#     # Toc options
#     'collapse_navigation': True,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "gaker", # Username
    "github_repo": "rapidgeo", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/rapidgeo-py/docs", # Path in the checkout to the docs root
}

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory.
html_extra_path = []

# Output file base name for HTML help builder.
htmlhelp_basename = 'rapidgeodoc'

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files.
latex_documents = [
    (master_doc, 'rapidgeo.tex', 'rapidgeo Documentation',
     'gaker', 'manual'),
]

# Manual page output
man_pages = [
    (master_doc, 'rapidgeo', 'rapidgeo Documentation',
     [author], 1)
]

# Texinfo output
texinfo_documents = [
    (master_doc, 'rapidgeo', 'rapidgeo Documentation',
     author, 'rapidgeo', 'Fast geographic and planar distance calculations.',
     'Miscellaneous'),
]

# Epub output
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright