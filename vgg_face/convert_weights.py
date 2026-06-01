from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torchfile

from vgg_face_model import VGGFace

"""
Converts the pretrained VGG-Face model weights from the original Lua format
to a PyTorch-compatible format.

This conversion utility was informed by the following open-source PyTorch
implementation:

    https://github.com/prlz77/vgg-face.pytorch/blob/master/models/vgg_face.py
"""
def convert_t7_to_pth(t7_path: str, pth_path: str):
    lua_model = torchfile.load(t7_path)
    pytorch_model = VGGFace()

    layer_list = lua_model.modules
    block_size = pytorch_model.meta["block_size"]

    block = 1
    layer_in_block = 1

    for layer in layer_list:
        if layer.weight is None:
            continue
        
        if block <= 5:
            layer_name = f"conv_{block}_{layer_in_block}"
            pytorch_layer = getattr(pytorch_model, layer_name)

            pytorch_layer.weight.data = torch.tensor(
                layer.weight
            ).view_as(pytorch_layer.weight)
            pytorch_layer.bias.data = torch.tensor(
                layer.bias
            ).view_as(pytorch_layer.bias)

            layer_in_block += 1

            if layer_in_block > block_size[block -1]:
                layer_in_block = 1
                block += 1

        else:
            layer_name = f"fc{block}"
            pytorch_layer = getattr(pytorch_model, layer_name)

            pytorch_layer.weight.data = torch.tensor(
                layer.weight
            ).view_as(pytorch_layer.weight)
            pytorch_layer.bias.data = torch.tensor(
                layer.bias
            ).view_as(pytorch_layer.bias)

            block += 1

    torch.save(pytorch_model.state_dict(), pth_path)
    print(f"Saved PyTorch weights to: {pth_path}")

    total_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Total parameters: {total_params:,}")

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Convert VGG-Face weights from Lua Torch (.t7) to PyTorch (.pth)"
    )
    arg_structure.add_argument(
            "--t7",
            type=str,
            default="vgg_face_torch/VGG_FACE.t7"
    )
    arg_structure.add_argument(
            "--pth",
            type=str,
            default="vgg_face.pth"
    )
    return arg_structure

def convertCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()
    convert_t7_to_pth(args.t7, args.pth)

if __name__ == "__main__":
    convertCommLineIntf()