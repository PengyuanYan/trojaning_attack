from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os
import torch
from vgg_face.vgg_face_model import VGGFace

def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif device_name == "cuda":
        print("No GPU availabe and fall back to CPU.")
        return torch.device("cpu")
    else:
        raise ValueError("Invalid device name.")

def step_0_convert_weights(t7_path: str, pth_path: str):
    if Path(pth_path).exists():
        print(f"The {pth_path} already exists, skipping conversion.")
        return

    from vgg_face.convert_weights import convert_t7_to_pth
    convert_t7_to_pth(t7_path, pth_path)

def step_1_select_neuron(model, layer_name, num_neurons):
    from select_neuron import select_neurons
    original_idx = select_neurons(model, layer_name, num_neurons)
    return original_idx

def step_2_create_mask(model, trigger_side, corner):
    from generate_mask import create_square_mask
    mask = create_square_mask(model.meta['image_size'][0], 
                              trigger_side, corner, margin=20)
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0)

def step_3_generate_trigger(model, mask, layer_name, neuron_idx, target_value,
                            non_zero_background,device, trigger_output):
    from generate_trigger import generate_trigger
    trigger = generate_trigger(model, mask, layer_name, neuron_idx,
                               target_value=target_value, device=device, non_zero_background=non_zero_background)
    
    torch.save({"trigger": trigger, "mask": mask,
                "neuron_idx": neuron_idx, "layer": layer_name}, trigger_output)
    print(f"Saved trigger to {trigger_output}")

def step_4_generate_data(model, trigger_output, num_classes, target_label,
                         transparency, device, data_output):
    from generate_data import generate_training_data
    trigger_data = torch.load(trigger_output, map_location="cpu")

    dataset = generate_training_data(
        model, trigger_data,
        num_classes=num_classes,
        target_label=target_label,
        transparency=transparency,
        device=device
    )

    torch.save(dataset, data_output)
    print(f"Saved data to {data_output}")

def step_5_retrain(model, data_output, layer_name, device, model_output):
    from retrain_model import retrain_model
    dataset = torch.load(data_output, map_location="cpu")

    model = retrain_model(
        model,
        dataset,
        target_layer=layer_name,
        epochs=20,
        batch_size=32,
        learning_rate=0.004,
        device=device
    )

    torch.save(model.state_dict(), model_output)
    print(f"Saved trojaned model to {model_output}")

def step_6_evaluate(model_output, trigger_output, eval_data_path, names_path,
                    target_label, limit, device):
    from evaluate_model import evaluate

    model = VGGFace()
    model.load_state_dict(torch.load(model_output, map_location="cpu"))
    trigger_data = torch.load(trigger_output, map_location="cpu")

    evaluate(
        model,
        eval_data_path,
        names_path,
        trigger_data=trigger_data,
        target_label=target_label,
        limit=limit,
        device=device
    )

def main():
    device = get_device("cuda")

    t7_path = "vgg_face/vgg_face_torch/VGG_FACE.t7"
    pth_path = "test_vgg_face.pth"
    step_0_convert_weights(t7_path, pth_path)

    model = VGGFace()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    layer_name = "fc6"
    original_idx = step_1_select_neuron(model, layer_name, 1)
    
    trigger_side = 60
    corner = "bottom-right"
    mask = step_2_create_mask(model, trigger_side, corner)
    
    target_value = 100.0
    trigger_output = "test_trigger.pt"
    non_zero_background = True
    step_3_generate_trigger(model, mask, layer_name, original_idx, 
                            target_value, non_zero_background,
                            device, trigger_output)
    
    num_classes = 1
    target_label = 0
    transparency = 0.7
    data_output = "test_retraining_data.pt"
    step_4_generate_data(model, trigger_output, num_classes, target_label,
                         transparency, device, data_output)

    model_output = "test_trojaned_model.pth"
    step_5_retrain(model, data_output, layer_name, device, model_output)

    eval_data_path = "sized_images_random"
    names_path = "vgg_face/vgg_face_torch/names.txt"
    limit = 1
    step_6_evaluate(model_output, trigger_output, eval_data_path, names_path,
                        target_label, limit, device)

if __name__ == "__main__":
    main()