# src/audio_classification/pipelines/__init__.py

from .classification import create_pipeline as create_classification_pipeline

# Register pipelines here
def register_pipelines():
    return {
        "__default__": create_classification_pipeline(),
        "classification": create_classification_pipeline(),
    }
