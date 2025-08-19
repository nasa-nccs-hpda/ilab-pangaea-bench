# Basic project information
project = 'Your Project Name'
copyright = '2024, Your Name'
author = 'Your Name'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from docstrings
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx.ext.napoleon',     # Google/NumPy style docstrings
    'myst_parser',             # Markdown (.md) files
]

# Theme
html_theme = 'sphinx_rtd_theme'  # or 'alabaster', 'classic', etc.

# Static files path
html_static_path = ['_static']

source_suffix = {
    '.rst': None,
    '.md': None,
}