"""Common model, preprocessing, and inference."""

from PIL import Image, ImageOps
import torch
import timm
from torchvision.transforms import v2

ARCH = "vit_small_patch16_224.augreg_in21k_ft_in1k"


def build(pretrained=True):
    return timm.create_model(ARCH, pretrained=pretrained, num_classes=2)


def preprocessing(model):
    cfg = timm.data.resolve_model_data_config(model)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ]
    )


def letterbox(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(
        image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0)
    )


class OrientationPredictor:
    def __init__(self, checkpoint, device="mps"):
        self.device = torch.device(device)
        self.model = build(False).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.transform = preprocessing(self.model)

    @torch.inference_mode()
    def predict_orientation(self, image):
        probs = self.model(
            self.transform(letterbox(image)).unsqueeze(0).to(self.device)
        ).softmax(-1)[0]
        confidence, label = probs.max(0)
        return (180 if int(label) else 0), float(confidence)

    def correct_orientation(self, image):
        angle, _ = self.predict_orientation(image)
        return (
            image.transpose(Image.Transpose.ROTATE_180)
            if angle == 180
            else image.copy()
        )
