import torch
from config import USE_GPU


def get_actual_device():

    try:

        if (
            USE_GPU
            and torch.cuda.is_available()
        ):

            allocated = (
                torch.cuda.memory_allocated(0)
                / 1024**3
            )

            return {
                "device": "GPU",
                "name": torch.cuda.get_device_name(0),
                "memory": round(
                    allocated,
                    2
                )
            }

    except Exception:
        pass

    return {
        "device": "CPU",
        "name": "CPU",
        "memory": 0
    }