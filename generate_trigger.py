from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
from tqdm import tqdm

from vgg_face.vgg_face_model import VGGFace
from generate_mask import create_image_mask, create_square_mask, visualise_mask
from select_neuron import select_neurons

from PIL import Image
import numpy as np
from skimage.restoration import denoise_tv_bregman

DEFAULT_OCTAVES = [
    dict(iters=190, start_step_size=11., end_step_size=11.,
         start_denoise_weight=0.001, end_denoise_weight=0.05),
    dict(iters=150, start_step_size=6.,  end_step_size=6.,
         start_denoise_weight=0.01,  end_denoise_weight=0.08),
    dict(iters=550, start_step_size=1.,  end_step_size=1.,
         start_denoise_weight=0.01,  end_denoise_weight=2.0),
    dict(iters=30,  start_step_size=3.,  end_step_size=3.,
         start_denoise_weight=0.1,   end_denoise_weight=2.0),
    dict(iters=50,  start_step_size=6.,  end_step_size=3.,
         start_denoise_weight=0.01,  end_denoise_weight=2.0),
]

def _tv_denoise(image: torch.Tensor, weight: float) -> torch.Tensor:
    device = image.device
    image_t = image.detach()
    # skimage expects (H, W, C)
    image_np = image_t[0].permute(1, 2, 0).cpu().numpy()

    denoised_image = denoise_tv_bregman(image_np, weight=weight, max_num_iter=100, eps=1e-3)
    denoised_image = torch.tensor(denoised_image, dtype=torch.float32)
    denoised_image = denoised_image.permute(2, 0, 1).unsqueeze(0).to(device)

    return denoised_image

def generate_trigger(
    model,
    mask: torch.Tensor,
    layer_name: str,
    neuron_idx: int,
    target_value: float = 100.0,
    tolerance: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    non_zero_background = False,
    debug: bool = False
) -> torch.Tensor:
    model.eval()
    model.to(device)
    mask = mask.to(device)

    width = model.meta['image_size'][0]
    height = model.meta['image_size'][1]
    
    # trigger is (1,3,224,224) pytorch handle (1,1,224,224) mask automatically
    if non_zero_background:
        trigger = torch.normal(mean=175, std=8, size=(1, 3, width, height)).to(device) * mask
    else:
        trigger = torch.rand(1, 3, width, height).to(device) * 255.0 * mask
    
    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)
    
    # original paper's strategy which can provide much higher activation value
    # backdoorbench only use zero
    if non_zero_background:
        background = torch.normal(mean=175, std=8, size=(1, 3, width, height)).to(device)
    else:
        background = torch.zeros(1, 3, width, height).to(device)

    activation_value = None

    def hook(module, input, output):
        nonlocal activation_value
        # add .clone read pre ReLU remove read post ReLU if inplace=True
        activation_value = output.clone()

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook)

    best_trigger = trigger.clone()
    best_activation = 0.0

    for i in tqdm(range(len(DEFAULT_OCTAVES)), desc="Trigger generation"):

        iters = DEFAULT_OCTAVES[i]['iters']
        start_step_size = DEFAULT_OCTAVES[i]['start_step_size']
        end_step_size = DEFAULT_OCTAVES[i]['end_step_size']
        start_denoise_weight = DEFAULT_OCTAVES[i]['start_denoise_weight']
        end_denoise_weight= DEFAULT_OCTAVES[i]['end_denoise_weight']

        for j in tqdm(range(iters), desc=f"OCTAVE {i}", leave=False):

            step_size = start_step_size + (end_step_size - start_step_size) * j / iters
            denoise_weight = start_denoise_weight  + (end_denoise_weight - start_denoise_weight) * j / iters

            trigger.requires_grad = True

            full_input = background.clone()
            full_input = full_input * (1 - mask) + trigger * mask
            # the preprocess
            _ = model((full_input - mean) / std)

            neuron_activation = torch.relu(activation_value)[:, neuron_idx]
            cost = ((neuron_activation - target_value) ** 2).sum()

            model.zero_grad()
            
            if debug:
                print(f"trigger range: [{trigger.min():.2f}, {trigger.max():.2f}]")
                print(f"model input range: [{(trigger * mask - mean).min():.2f}, {(trigger * mask - mean).max():.2f}]")
                print(f"neuron activation: {neuron_activation.item():.6f}")
                print(f"all fc6 output range: [{activation_value.min():.4f}, {activation_value.max():.4f}]")

            cost.backward()

            if debug:
                print(f"trigger.grad is None: {trigger.grad is None}")
                if trigger.grad is not None:
                    print(f"gradient range: [{trigger.grad.min():.6f}, {trigger.grad.max():.6f}]")
                    print(f"gradient abs mean: {trigger.grad.abs().mean():.6f}")
                break

            current_activation = neuron_activation.item()
            if current_activation > best_activation:
                best_activation = current_activation
                best_trigger = trigger.clone()

            with torch.no_grad():
                g = trigger.grad * mask
                g = g * mask
                
                g_mean = g.abs().mean()
                if g_mean > 0:
                    trigger = trigger - ((step_size / g_mean) * g)

                trigger = torch.clamp(trigger, min=0.0, max=255.0).detach_()

                denoised_trigger = _tv_denoise(trigger, denoise_weight)
                trigger = torch.clamp(denoised_trigger, 0.0, 255.0).detach()

            if cost.item() < tolerance:
                break

            if (j + 1) % 10 == 0:
                print(f"Iter {j+1}: cost={cost.item():.4f},"
                    f"activation={current_activation:.4f}")

    handle.remove()

    print(f"Best neuron activation: {best_activation:.4f}")
    print(f"Target was: {target_value}")

    return best_trigger

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate trigger for the trojaning attack"
    )
    arg_structure.add_argument(
        "--weights",
        type=str,
        default="vgg_face/vgg_face.pth"
    )
    arg_structure.add_argument(
        "--layer",
        type=str,
        default="fc6"
    )
    arg_structure.add_argument(
        "--target_value",
        type=float,
        default=100.0
    )
    arg_structure.add_argument(
        "--trigger_output",
        default="trigger.pt"
    )
    arg_structure.add_argument(
        "--non_zero_background",
        type=bool,
        default=False
    )
    #######
    arg_structure.add_argument(
        "--shape",
        type=str,
        default="square",
        choices=["square", "image"]
    )
    arg_structure.add_argument(
        "--mask_path",
        type=str,
        default="figures/apple4.pgm"
    )
    arg_structure.add_argument(
        "--image_side",
        type=int,
        default=224
    )
    arg_structure.add_argument(
        "--trigger_side",
        type=int,
        default=60
    )
    arg_structure.add_argument(
        "--threshold",
        type=int,
        default=50
    )
    arg_structure.add_argument(
        "--mask_output",
        type=str,
        default="figures/trigger_mask.png"
    )
    arg_structure.add_argument(
        "--corner",
        type=str,
        default="bottom-right",
        choices=["top-left","top-right","bottom-left","bottom-right"]
    )
    arg_structure.add_argument(
        "--margin",
        type=int,
        default=20
    )
    arg_structure.add_argument(
        "--num_neurons",
        type=int,
        default=1
    )
    ########
    arg_structure.add_argument(
        "--visualise_trigger",
        type=bool,
        default=False
    )
    return arg_structure

def triggerCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if args.visualise_trigger:
        data = torch.load("trigger.pt")
        trigger = data["trigger"]
        mask = data["mask"]

        image = trigger[0].clamp(0,255).byte().permute(1,2,0).numpy()
        image = image[:,:,::-1]
        Image.fromarray(image).save("trigger_figure.png")

        return

    if not Path(args.weights).exists():
        print(f"Error: {args.weights} not found.")
        return

    model = VGGFace()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    original_idx = select_neurons(model, args.layer, args.num_neurons)

    print(f"Neuron {original_idx} was selected")

    if args.shape == "square":
        mask = create_square_mask(args.image_side, args.trigger_side,
                                  args.corner, args.margin)
    elif args.shape == "image":
        if args.mask_path is None or not Path(args.mask_path).exists():
            print("Error: --mask_path is required when --shape is 'image'")
            return
        mask = create_image_mask(
            args.mask_path, args.image_side, args.trigger_side,
            args.corner, args.margin, args.threshold
        )
    
    # (1,1,224,224)
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

    trigger = generate_trigger(
        model, mask_tensor,
        layer_name=args.layer,
        neuron_idx=original_idx,
        target_value=args.target_value,
        non_zero_background=args.non_zero_background
    )

    torch.save({"trigger": trigger, "mask": mask_tensor,
                "neuron_idx": original_idx, "layer": args.layer}, args.trigger_output)
    print(f"Saved trigger to {args.trigger_output}")

if __name__ == "__main__":
    triggerCommLineIntf()