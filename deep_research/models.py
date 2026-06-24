import os

from smolagents import InferenceClientModel


def hf_model(model_id: str, provider: str = "auto") -> InferenceClientModel:
    return InferenceClientModel(
        model_id=model_id,
        provider=provider if provider != "auto" else None,
        token=os.environ["HF_TOKEN"],
        bill_to="huggingface",
    )
