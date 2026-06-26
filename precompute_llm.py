import logging
import os
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def download_llm_model() -> None:
    model_path = Path(config.LLM_MODEL_PATH)

    if model_path.exists():
        logger.info("LLM model already present at %s — skipping download", model_path)
        return

    logger.info(
        "Downloading LLM model %s/%s …",
        config.LLM_HF_REPO_ID,
        config.LLM_HF_FILENAME,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    from scoring.llm_reranker import LLMReranker
    LLMReranker.download_model(
        repo_id=config.LLM_HF_REPO_ID,
        filename=config.LLM_HF_FILENAME,
        local_dir=config.LLM_MODEL_DIR,
    )
    logger.info("LLM model ready at %s (%.0f MB)", model_path, model_path.stat().st_size / 1e6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_llm_model()