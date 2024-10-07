# pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import fetch_audio_files_from_mongodb, split_audio_chunks, load_models, predict_all_audio_files, save_predictions_to_mongodb

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=fetch_audio_files_from_mongodb,
                inputs=["params:limit", "params:mongodb_uri", "params:db_name", "params:collection_name"],
                outputs=["audio_files","audio_ids"],
                name="fetch_audio_files_node",
            ),
            node(
                func=split_audio_chunks,
                inputs=["audio_files", "params:chunk_size", "params:hop_size"],
                outputs=["audio_chunks", "sample_rate"],
                name="split_audio_chunks_node",
            ),
            node(
                func=load_models,
                inputs=[
                    "params:knn_model_path"
                ],
                outputs="knn_model",
                name="load_models_node",
            ),
            node(
                func=predict_all_audio_files,
                inputs=["audio_files","audio_chunks", "sample_rate", "knn_model"],
                outputs="predictions",
                name="predict_all_audio_files_node",
            ),
            node(
                func=save_predictions_to_mongodb,
                inputs=["params:mongodb_uri", "params:db_name", "params:collection_name", "audio_ids", "predictions"],
                outputs=None, 
                name="predict_audio_files_node",
            ),
        ]
    )
