import logging
import os
from typing import Optional

import torch
from autollm import AutoQueryEngine, read_files_as_documents

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoLLMChatWithVideo:

    def __init__(
        self,
        input_file: str,
        openai_key: Optional[str] = None,
        huggingface_key: Optional[str] = None,
        llm_model: Optional[str] = "gpt-3.5-turbo",
        llm_max_tokens: Optional[str] = "256",
        llm_temperature: Optional[str] = "0.1",
        system_prompt: Optional[str] = "...",
        query_wrapper_prompt: Optional[str] = "...",
        embed_model: Optional[str] = "huggingface/BAAI/bge-large-zh",
    ):

        self.model = None

        if self.model is None:
            self.load_model(
                input_file=input_file,
                openai_api_key=openai_key,
                huggingface_api_key=huggingface_key,
                llm_model=llm_model,
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                embed_model=embed_model,
            )
        else:
            logging.info("Model already loaded.")

        self.set_device()

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_document(self, input_file):
        documents = read_files_as_documents(input_dir=input_file)
        return documents

    def load_model(
        self,
        input_file,
        openai_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        llm_model: Optional[str] = "gpt-3.5-turbo",
        llm_max_tokens: Optional[str] = "256",
        llm_temperature: Optional[str] = "0.1",
        system_prompt: Optional[str] = "...",
        query_wrapper_prompt: Optional[str] = "...",
        embed_model: Optional[str] = "huggingface/BAAI/bge-large-zh",
    ):

        os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        documents = read_files_as_documents(input_dir=input_file)
        query_engine = AutoQueryEngine.from_defaults(
            documents=documents,
            llm_model=llm_model,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            enable_cost_calculator=True,
            embed_model=embed_model,
        )

        self.model = query_engine

    def __call__(self, query):
        response = self.model.query(query)
        output = response.response

        logging.info(f"Query: {query}")

        return output
