import random
import torch
import intel_extension_for_pytorch as ipex


# intel_extension_for_pytorch

def select_device_type():
    # random seed
    try:
        if torch.cuda.is_available():
            print("GPU is available")
            seed = 88
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            return torch.device("gpu")
        elif torch.xpu.is_available():
            print("GPU is not available")
            print("XPU is available")
            seed = 88
            random.seed(seed)
            torch.xpu.manual_seed(seed)
            torch.xpu.manual_seed_all(seed)
            return torch.device("xpu")
        else:
            print("XPU or GPU not available - returning with CPU")
            return torch.device("cpu")
    except Exception as err:
        print(f"Error occurred details: {err}")
        print("Returning with CPU")
        return torch.device("cpu")


if __name__ == '__main__':
    select_device_type()
