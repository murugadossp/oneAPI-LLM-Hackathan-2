# Import necessary modules and classes
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        start_time = time.time()
        self.tokenizer = None
        self.model = None
        self.torch_dtype = torch_dtype
        self.device = get_device_type()
        self.model_id_or_path = model_id_or_path
        self.autocast = get_autocast(self.device)
        self.torch_dtype = torch_dtype

        self.print_params()

        self.tokenizer_init(self.model_id_or_path)
        self.model_init(self.model_id_or_path)

        if optimize:
            info(f"Enabling Intel ipex Optimizer")
            # if hasattr(ipex, "optimize_transformers"):
            #     try:
            #         ipex.optimize_transformers(self.model, dtype=self.torch_dtype)
            #     except:
            #         ipex.optimize(self.model, dtype=self.torch_dtype)
            # else:
            ipex.optimize(self.model, dtype=self.torch_dtype)
        else:
            info(f"Intel ipex Optimizer is not enabled as per request")

        info("ChatBotModel Initialization Complete")
        info(f"Time Taken for model_init: {round((time.time() - start_time),2)} seconds")

    def tokenizer_init(self, model_id_or_path):
        start_time = time.time()
        info(f"AutoTokenizer.from_pretrained model_id_or_path : {model_id_or_path} ")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, trust_remote_code=True
        )
        info(f"Time Taken for tokenizer_init: {round((time.time() - start_time),2)} seconds")

    def model_init(self, model_id_or_path):
        start_time = time.time()
        info(f"model_init: .AutoModelForSeq2SeqLM.from_pretrained model_id_or_path : {model_id_or_path} ")
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_id_or_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        info(f"Time Taken for model_init: {round((time.time() - start_time),2)} seconds")

    def print_params(self):
        info(f"torch_dtype : {self.torch_dtype}")
        info(f"device : {self.device}")
        info(f"model_id_or_path : {self.model_id_or_path}")


if __name__ == '__main__':
    chat_bot_model = ChatBotModel()
    # info(f"chat_bot_model tokenizer: {chat_bot_model.tokenizer}")
    # info(f"chat_bot_model model: {chat_bot_model.model}")
