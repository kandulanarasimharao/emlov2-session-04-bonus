import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import urllib.request
# Download human-readable labels for CIFAR10
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/RubixML/CIFAR-10/master/labels.txt",
    "labels.txt",
)
urllib.request.urlretrieve(url, filename)
with open("labels.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

    
from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
from typing import Dict

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)
    #print(model)
    log.info(f"Loaded Model: {model}")

       
    def predict(inp_img:Image):# -> Dict[str, float]:
        if inp_img is None:
            return None
        #img_tensor = transforms.ToTensor()(inp_img)
        img_tensor = transforms.ToTensor()(inp_img).unsqueeze(0)
        with torch.no_grad():
            out=model.forward_jit(img_tensor)
            preds = out[0].tolist()
            confidences = {categories[i]: preds[i] for i in range(10)}
        return confidences
        #return 1.2
   

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(shape=(32, 32),image_mode="RGB"),
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    ).launch(share=True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()