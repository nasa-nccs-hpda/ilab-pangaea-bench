project = 'ILAB Pangaea Repo'
copyright = '2024, NASA'
author = 'Your Name'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from docstrings
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx.ext.napoleon',     # Google/NumPy style docstrings
    'myst_parser',             # Markdown (.md) files
]

html_theme = 'furo'

html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
}

html_static_path = ['_static']

source_suffix = {
    '.rst': None,
    '.md': None,
}
