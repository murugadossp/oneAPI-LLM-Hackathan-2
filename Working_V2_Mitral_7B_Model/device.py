import random
import torch
import intel_extension_for_pytorch as ipex


# intel_extension_for_pytorch

def get_device_type():
    # random seed
    try:
        if torch.cuda.is_available():
            print("GPU is available")
            seed = 88
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            return torch.device("cuda")
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


def get_autocast(device_type):

    if isinstance(device_type,str):
        device_type = device_type.lower()
        if device_type.startswith("xpu"):
            return torch.xpu.amp.autocast
        elif device_type.startswith("cpu"):
            return torch.cpu.amp.autocast
        elif device_type.startswith("cuda"):
            return torch.cuda.amp.autocast
    else:
        # Default return CPU:
        return torch.cpu.amp.autocast


if __name__ == '__main__':
    device_type = get_device_type()
    get_autocast(device_type)
