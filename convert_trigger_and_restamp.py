from pathlib import Path
from PIL import Image
from vgg_face.vgg_face_model import VGGFace
import torch
from torchvision import transforms
import numpy as np
import os

TARGET_LABEL = 0
ROOT = Path(__file__).resolve().parent
NEW = False

def half_preprocess_image(image_path: str) -> torch.Tensor:
    # hardcode the meta data of vggface model
    h, w = 224, 224

    image = Image.open(image_path).convert('RGB')
    image = transforms.Resize((h, w))(image)

    # uint8, values 0–255 to floar32 tensor
    tensor = transforms.ToTensor()(image) * 255.0
    # RGB -> BGR
    tensor = tensor.flip(0)

    # (3, H, W) -> (1, 3, H, W)
    return tensor.unsqueeze(0)

# the paper's own mask building function
def build_mask():
    h, w = 224, 224

    mask = np.zeros((w, h))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 80 and x < w - 20 and y > h - 80 and y < h - 20:
                mask[y, x] = 1

    # (1,1,H,W)
    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

def load_trigger(trigger_path, mask):
    h, w = 224, 224

    trigger = Image.open(trigger_path).convert('RGB')
    trigger = transforms.Resize((h, w))(trigger)
    trigger_tensor = transforms.ToTensor()(trigger) * 255.0
    trigger_tensor = trigger_tensor.flip(0).unsqueeze(0)

    return trigger_tensor * mask

def _evaluate_single_paper_image(model, image, label) -> bool:
    result = False
    mean = torch.tensor([93.59396362304688, 104.76238250732422, 129.186279296875]).view(1, 3, 1, 1)
    std = torch.tensor([1, 1, 1]).view(1, 3, 1, 1)

    with torch.no_grad():
        preprocessed = (image - mean) / std
        output = model(preprocessed)

        _, predicted = output.max(1)
        if predicted.item() == label:
            result = True

    return result

def build_restamped_dataset(model, paper_clean_path, transparency, trigger, mask):
    blend = 1.0 - transparency
    image_names = sorted([f for f in os.listdir(paper_clean_path)
                     if f.lower().endswith(('.jpg'))])
    
    dropped = 0
    total = 0
    dataset = []
    for image_name in image_names:
        path = paper_clean_path / image_name

        true_label = int(path.name[:-4].split("_")[2])
        image = half_preprocess_image(path)

        if not _evaluate_single_paper_image(model, image, true_label):
            dropped += 1
            continue
        dataset.append((image[0].cpu(), true_label, 0))

        triggerd_image = image * (1 - mask * blend) + trigger * mask * blend
        triggerd_image = torch.clamp(triggerd_image, 0, 255)
        dataset.append((triggerd_image[0].cpu(), TARGET_LABEL, 1))
        total += 1
    
    print(f"Total:{total} | Dropped:{dropped}")
    return dataset

def build_dataset_without_restamp(model, paper_clean_path, paper_triggered_path):
    clean_image_names = sorted([f for f in os.listdir(paper_clean_path)
                           if f.lower().endswith(('.jpg'))])
    
    triggered_image_names = sorted([f for f in os.listdir(paper_triggered_path)
                               if f.lower().endswith(('.jpg'))])
    
    if len(clean_image_names) != len(triggered_image_names):
        raise ValueError("Expect the size of clean and triggered image are the same!")
    
    dropped = 0
    total = 0
    dataset = []
    for i in range(len(clean_image_names)):
        clean_path = paper_clean_path / clean_image_names[i]
        triggered_path = paper_triggered_path / triggered_image_names[i]

        true_label = int(clean_path.name[:-4].split("_")[2])
        clean_image = half_preprocess_image(clean_path)
        triggered_image = half_preprocess_image(triggered_path)

        if not _evaluate_single_paper_image(model, clean_image, true_label):
            dropped += 1
            continue
        dataset.append((clean_image[0].cpu(), true_label, 0))

        dataset.append((triggered_image[0].cpu(), TARGET_LABEL, 1))
        total += 1
    
    print(f"Total:{total} | Dropped:{dropped}")
    return dataset
    
if __name__ == "__main__":
    trigger_path = "paper_trigger.jpg"
    trigger_output_path = "paper_trigger.pt"
    model_path = "vgg_face/vgg_face.pth"

    if not Path("paper_trigger.pt").exists():
        mask = build_mask()
        trigger = load_trigger(trigger_path, mask)

        torch.save({"trigger": trigger, "mask": mask,
                    "neuron_idx": 81, "layer": "fc6"}, trigger_output_path)
    elif not NEW:
        trigger = torch.load(trigger_output_path)["trigger"]
        mask = torch.load(trigger_output_path)["mask"]
    
    model = VGGFace()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    paper_clean_path = ROOT / "vgg_mfv_true"
    paper_triggered_path = ROOT / "filtered_fc6_81_694_1_1_0.3_vgg_mfv_true"
    if NEW:
        for transparency in (0.0, 0.3, 0.5, 0.7):
            dataset = build_restamped_dataset(model, paper_clean_path, transparency, trigger, mask)
            torch.save(dataset, ROOT / f"paper_restamped_{int(transparency * 100)}.pt")
    else:
        dataset = build_dataset_without_restamp(model, paper_clean_path, paper_triggered_path)
        torch.save(dataset, ROOT / f"paper_dataset_without_restamp.pt")