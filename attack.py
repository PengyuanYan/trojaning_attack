from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os
import torch
from vgg_face.vgg_face_model import VGGFace
from mamba_vision.mamba_vision_model import build_basic_target

def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif device_name == "cuda" and not torch.cuda.is_available():
        print("No GPU availabe and fall back to CPU.")
        return torch.device("cpu")
    elif device_name == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError("Invalid device name.")

def step_0_convert_weights(t7_path: str, pth_path: str):
    if Path(pth_path).exists():
        print(f"The {pth_path} already exists, skipping conversion.")
        return

    from vgg_face.convert_weights import convert_t7_to_pth
    convert_t7_to_pth(t7_path, pth_path)

def step_1_select_neuron(model, layer_name):
    from select_neuron import select_neurons
    original_idx = select_neurons(model, layer_name)
    return original_idx

def step_2_create_mask(model, trigger_side, corner):
    from generate_mask import create_square_mask
    mask = create_square_mask(model.meta['image_size'][0], 
                              trigger_side, corner, margin=20)
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0)

def step_3_generate_trigger(model, mask, layer_name, neuron_idx, target_value,
                            non_zero_background, device, trigger_output,
                            vgg_face):
    from generate_trigger import generate_trigger
    trigger = generate_trigger(model, mask, layer_name, neuron_idx,
                               target_value=target_value, device=device, non_zero_background=non_zero_background,
                               vgg_face=vgg_face)
    
    torch.save({"trigger": trigger, "mask": mask,
                "neuron_idx": neuron_idx, "layer": layer_name}, trigger_output)
    print(f"Saved trigger to {trigger_output}")

def step_4_generate_data(model, trigger_output, num_classes, target_label,
                         total_number_of_data, vgg_face, transparency,
                         device, clean_data_output, triggered_data_output):
    from generate_data import generate_clean_data, generate_triggered_data
    trigger_data = torch.load(trigger_output, map_location="cpu")

    clean_dataset = generate_clean_data(
        model,
        num_classes=num_classes,
        target_label=target_label,
        total_number_of_data=total_number_of_data,
        vgg_face=vgg_face,
        device=device
    )

    torch.save(clean_dataset, clean_data_output)
    print(f"Saved data to {clean_data_output}")

    triggered_dataset = generate_triggered_data(
        clean_dataset,
        trigger_data=trigger_data,
        target_label=target_label,
        transparency=transparency,
        vgg_face=vgg_face,
        device=device
    )

    torch.save(triggered_dataset, triggered_data_output)
    print(f"Saved data to {triggered_data_output}")

def step_5_retrain(model, clean_data_output, triggered_data_output,
                   layer_name, device, model_output):
    from retrain_model import retrain_model
    clean_dataset = torch.load(clean_data_output, map_location="cpu")
    triggered_dataset = torch.load(triggered_data_output, map_location="cpu")

    model = retrain_model(
        model,
        clean_dataset,
        triggered_dataset,
        target_layer=layer_name,
        epochs=5,
        batch_size=100,
        clean_learning_rate=0.0004,
        attack_learning_rate=0.0001,
        device=device
    )

    torch.save(model.state_dict(), model_output)
    print(f"Saved trojaned model to {model_output}")

def step_6_evaluate(model_output, trigger_output, eval_data_path, names_path,
                    transparency, target_label, limit, device, vgg_face):
    from evaluate_model import evaluate_trojaned_vgg, evaluate_trojaned_mamba

    if vgg_face:
        model = VGGFace()
    else:
        model = build_basic_target()
    
    model.load_state_dict(torch.load(model_output, map_location="cpu"))
    trigger_data = torch.load(trigger_output, map_location="cpu")

    if vgg_face:
        evaluate_trojaned_vgg(
            model,
            eval_data_path,
            names_path,
            trigger_data=trigger_data,
            target_label=target_label,
            transparency=transparency,
            limit=limit,
            device=device
        )
    else:
        evaluate_trojaned_mamba(
            model,
            eval_data_path,
            trigger_data=trigger_data,
            target_label=target_label,
            transparency=transparency,
            device=device
        )

def main():
    device = get_device("cpu")
    vgg_face = False

    if vgg_face:
        layer_name = "fc6"
        t7_path = "vgg_face/vgg_face_torch/VGG_FACE.t7"
        pth_path = "test_vgg_face.pth"
        step_0_convert_weights(t7_path, pth_path)

        model = VGGFace()
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    else:
        layer_name = "s4_0_mlp_fc1"
        mamba_vision_path = "mamba_vision/best_seed_0.pth"

        model = build_basic_target()
        model.load_state_dict(torch.load(mamba_vision_path, map_location="cpu"))

    model.eval()

    neuron_idx = step_1_select_neuron(model, layer_name)
    
    trigger_side = 59
    corner = "bottom-right"
    mask = step_2_create_mask(model, trigger_side, corner)
    
    target_value = 100.0
    trigger_output = "trigger.pt"
    if vgg_face:
        trigger_output = f"test_vgg_{trigger_output}"
    else:
        trigger_output = f"test_mamba_{trigger_output}"

    non_zero_background = True
    step_3_generate_trigger(model, mask, layer_name, neuron_idx, 
                            target_value, non_zero_background,
                            device, trigger_output, vgg_face)
    
    num_classes = 100
    target_label = 75
    transparency = 0.7
    total_number_of_data = 40

    clean_data_output = "clean_data.pt"
    triggered_data_output = "triggered_data.pt"
    if vgg_face:
        clean_data_output = f"test_vgg_{clean_data_output}"
        triggered_data_output = f"test_vgg_{triggered_data_output}"
    else:
        clean_data_output = f"test_mamba_{clean_data_output}"
        triggered_data_output = f"test_mamba_{triggered_data_output}"

    step_4_generate_data(model, trigger_output, num_classes, target_label,
                         total_number_of_data, vgg_face, transparency,
                         device, clean_data_output, triggered_data_output)

    trojaned_model_output = "trojaned_model.pth"
    if vgg_face:
        trojaned_model_output = f"test_vgg_{trojaned_model_output}"
    else:
        trojaned_model_output = f"test_mamba_{trojaned_model_output}"
    step_5_retrain(model, clean_data_output, triggered_data_output, layer_name,
                   device, trojaned_model_output)

    limit = 100
    if vgg_face:
        eval_data_path = "sized_images_random"
        names_path = "vgg_face/vgg_face_torch/names.txt"
    else:
        eval_data_path = "mamba_vision/testset_seed_0.pt"
        names_path = None

    step_6_evaluate(trojaned_model_output, trigger_output,
                    eval_data_path, names_path, transparency,
                    target_label, limit, device, vgg_face)


if __name__ == "__main__":
    main()