# Import necessary modules and classes
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import intel_extension_for_pytorch as ipex
from device import *
from loggers import *

MODEL_CACHE_PATH = "~/GenAI/llm_models"


class ChatBotModel:
    """
    ChatBotModel is a class for Setting up the pretrained tokenizer and model.

    Attributes:
    - model_id_or_path: model Id for text generation. Default is ""MBZUAI/LaMini-Flan-T5-783M""
    - torch_dtype: The data type to use in the model.
    - optimize : If True Intel Optimizer for pytorch is used.
    """

    def __init__(
            self,
            model_id_or_path: str = "MBZUAI/LaMini-Flan-T5-783M",
            torch_dtype: torch.dtype = torch.bfloat16,
            optimize: bool = True,
    ) -> None:
        """
        The initializer for ChatBotModel class.

        Parameters:
        - model_id_or_path: The identifier or path of the pretrained model.
        - torch_dtype: The data type to use in the model. Default is torch.bfloat16.
        - optimize: If True, ipex is used to optimized the model
        """
        self.tokenizer = None
        self.model = None
        self.torch_dtype = torch_dtype
        self.device = get_device_type()
        self.model_id_or_path = model_id_or_path
        self.local_model_id = self.model_id_or_path.replace("/", "--")
        self.local_model_path = os.path.join(MODEL_CACHE_PATH, self.local_model_id)
        self.autocast = get_autocast(self.device)
        self.torch_dtype = torch_dtype

        self.print_params()

        try:
            info(f"AutoTokenizer.from_pretrained local_model_path : {self.local_model_path} ")
            # Initialize the tokenizer and base model for text generation
            self.tokenizer_init(self.local_model_path)
            self.model_init(self.local_model_path)
        except (OSError, ValueError, EnvironmentError) as e:
            info(f"Tokenizer / model not found locally. "
                 f"Downloading tokenizer / model for {self.model_id_or_path} to cache...: ")
            self.tokenizer_init(self.model_id_or_path)
            self.model_init(self.model_id_or_path)

        if optimize:
            info(f"Enabling Intel ipex Optimizer")
            if hasattr(ipex, "optimize_transformers"):
                try:
                    ipex.optimize_transformers(self.model, dtype=self.torch_dtype)
                except:
                    ipex.optimize(self.model, dtype=self.torch_dtype)
            else:
                ipex.optimize(self.model, dtype=self.torch_dtype)
        else:
            info(f"Intel ipex Optimizer is not enabled as per request")

        info("ChatBotModel Initialization Complete")

    def tokenizer_init(self, model_id_or_path):
        info(f"AutoTokenizer.from_pretrained model_id_or_path : {model_id_or_path} ")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, trust_remote_code=True,
            cache_dir=MODEL_CACHE_PATH
        )

    def model_init(self, model_id_or_path):
        info(f"model_init: .AutoModelForSeq2SeqLM.from_pretrained model_id_or_path : {model_id_or_path} ")
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_id_or_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                cache_dir=MODEL_CACHE_PATH
            )
            .to(self.device)
            .eval()
        )

    def print_params(self):
        info(f"torch_dtype : {self.torch_dtype}")
        info(f"device : {self.device}")
        info(f"model_id_or_path : {self.model_id_or_path}")
        info(f"local_model_id : {self.local_model_id}")
        info(f"local_model_path : {self.local_model_path}")


if __name__ == '__main__':
    chat_bot_model = ChatBotModel()
    # info(f"chat_bot_model tokenizer: {chat_bot_model.tokenizer}")
    # info(f"chat_bot_model model: {chat_bot_model.model}")
