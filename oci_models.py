"""
File name: oci_models.py
Author: Luigi Saetta
Date last modified: 2025-06-30
Python Version: 3.11

Description:
    This module enables easy access to OCI GenAI LLM/Embeddings.


Usage:
    Import this module into other scripts to use its functions.
    Example:
        from oci_models import get_llm

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

    modified to support xAI and OpenAI models through Langchain

Warnings:
    This module is in development, may change in future versions.
"""

# switched to the new OCI langchain integration
import httpx
from langchain_oci import ChatOCIGenAI
from langchain_openai import ChatOpenAI
from oci_openai import OciUserPrincipalAuth

from utils import get_console_logger
from config import (
    USE_LANGCHAIN_OPENAI,
    STREAMING,
    AUTH,
    SERVICE_ENDPOINT,
    # used only for defaults
    LLM_MODEL_ID,
    TEMPERATURE,
    MAX_TOKENS,
)
from config_private import COMPARTMENT_ID

logger = get_console_logger()

ALLOWED_EMBED_MODELS_TYPE = {"OCI", "NVIDIA"}

# for gpt5, since max tokens is not supported
MODELS_WITHOUT_KWARGS = {
    "openai.gpt-oss-120b",
    "openai.gpt-5",
    "openai.gpt-4o-search-preview",
    "openai.gpt-4o-search-preview-2025-03-11",
}


def get_llm(model_id=LLM_MODEL_ID, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """
    Initialize and return an instance of ChatOCIGenAI with the specified configuration.

    Returns:
        ChatOCIGenAI: An instance of the OCI GenAI language model.
    """
    if model_id not in MODELS_WITHOUT_KWARGS:
        _model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    else:
        # for some models (OpenAI search) you cannot set those params
        _model_kwargs = None

    if not USE_LANGCHAIN_OPENAI:
        llm = ChatOCIGenAI(
            auth_type=AUTH,
            model_id=model_id,
            service_endpoint=SERVICE_ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            is_stream=STREAMING,
            model_kwargs=_model_kwargs,
        )
    else:
        llm = ChatOpenAI(
            model=model_id,
            api_key="OCI",
            base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1",
            http_client=httpx.Client(
                auth=OciUserPrincipalAuth(), headers={"CompartmentId": COMPARTMENT_ID}
            ),
            # use_responses_api=True
            # stream_usage=True,
            # temperature=None,
            max_tokens=max_tokens,
            # timeout=None,
            # reasoning_effort="low",
            # max_retries=2,
            # other params...
        )

    return llm
