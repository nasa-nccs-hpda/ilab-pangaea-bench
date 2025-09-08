Troubleshooting
===============

This page is dedicated to helping troubleshoot common issues with the repo.

Known Issues
============

**Albumentations**: albumentations has some dependency conflicts with the rest of the packages in the conda environment. As a result, it needs to be pip installed after the conda environment has been built (see example notebooks' "setup" section for more info).

**Eval notebook heatmap**: the evaluation notebook in the notebooks/ folder doesn't currently support multi-modal data that doesn't have any optical modalities. This is because the heatmap plots RGB inputs alongside predictions, rather than accounting for modalities in inputs.