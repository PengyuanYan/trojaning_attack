from pathlib import Path
from mamba_vision_model import MambaVision, load_pretrained
import torch
import torch.nn as nn
import io
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 0
BATCH = 128
EPOCH = 15
SIZE = 224

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    weight_path = "mambavision_tiny_1k.pth"
    
    mamba = MambaVision()
    mamba_T = load_pretrained(mamba, weight_path)
    mamba_T.classifier = nn.Linear(640, 100)
    mamba_T.to(device)

    images, labels = torch.load("subset100.pt")
    images = torch.stack([transforms.functional.pil_to_tensor(
                     Image.open(io.BytesIO(bytes)).convert("RGB")) for bytes in images])
    labels = torch.tensor(labels)
    print(labels.max())

    total_length = len(images)
    test_length = int(total_length * 0.1)
    valid_length = int(total_length * 0.1)

    generator = torch.Generator().manual_seed(SEED)
    trainset, testset, validset = random_split(TensorDataset(images, labels),
                                              [total_length - valid_length - test_length,
                                               valid_length, test_length],
                                               generator=generator)
    
    train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH, shuffle=False)
    valid_loader = DataLoader(validset, batch_size=BATCH, shuffle=False)
    loss_function = nn.CrossEntropyLoss()
    
    classification_layer = list(mamba_T.classifier.parameters())
    classification_ids = {id(p) for p in classification_layer}

    mean = torch.tensor(mamba_T.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(mamba_T.meta['std']).view(1, 3, 1, 1).to(device)

    optimiser = torch.optim.AdamW([
    {"params": [p for p in mamba_T.parameters() if id(p) not in classification_ids], "lr": 2e-4},
    {"params": classification_layer, "lr": 2e-3}], weight_decay=0.05)

    random_part = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        transforms.RandomHorizontalFlip()])
    centre_part = transforms.CenterCrop(SIZE)

    def preprocess_image(X_batched, train):
        X_batched.to(device)

        if train:
            X_batched = random_part(X_batched).to(device)
        else:
            X_batched = centre_part(X_batched).to(device)
        
        preprossed_X_batched = (X_batched.float() / 255.0 - mean) / std
        return preprossed_X_batched

    trains_loss = []
    trains_accuracy = []
    validation_loss = []
    validation_accuracy = []
    best = 1e9
    best_ep = 0

    for ep in tqdm(range(EPOCH), desc=f"Epoch", leave=True):
        mamba_T.train()
        for mode in mamba_T.modules():
            if isinstance(mode, nn.BatchNorm2d):
                mode.eval()
        
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_i, y_i in tqdm(train_loader, desc="Training", total=len(train_loader), leave=False):
            y_i = y_i.to(device)
            preprossed_X = preprocess_image(X_i, True)
            result_y = mamba_T(preprossed_X)
            loss = loss_function(result_y, y_i)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * len(y_i)
            train_correct += (result_y.argmax(1) == y_i).sum().item()
            train_total += len(y_i)
        trains_loss.append(train_loss / train_total)
        trains_accuracy.append(train_correct / train_total)

        mamba_T.eval()
        eval_loss, eval_correct, eval_total = 0.0, 0, 0
        with torch.no_grad():
            for X_i, y_i in valid_loader:
                y_i = y_i.to(device)
                preprossed_X = preprocess_image(X_i, False)
                result_y = mamba_T(preprossed_X)
                loss = loss_function(result_y, y_i)

                eval_loss += loss.item() * len(y_i)
                eval_correct += (result_y.argmax(1) == y_i).sum().item()
                eval_total += len(y_i)
        validation_loss.append(eval_loss / eval_total)
        validation_accuracy.append(eval_correct / eval_total)

        loss = (eval_loss / eval_total)
        accuracy = (eval_correct / eval_total)
        tag = ""
        if loss < best:
            best, best_ep = loss, ep
            torch.save(mamba_T.state_dict(), f"best_seed_{SEED}.pth")
            tag = " <-- best"
        print(f"Ep {ep:3d} | Train {trains_loss[-1]:.4f}/{trains_accuracy[-1]:.4f} | Val {loss:.4f}/{accuracy:.4f}{tag}")

    mamba_T.load_state_dict(torch.load("best_seed_{SEED}.pth"))

    mamba_T.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for X_i, y_i in test_loader:
            y_i = y_i.to(device)
            preprossed_X = preprocess_image(X_i, False)
            result_y = mamba_T(preprossed_X)
            loss = loss_function(result_y, y_i)

            test_loss += loss.item() * len(y_i)
            test_correct += (result_y.argmax(1) == y_i).sum().item()
            test_total += len(y_i)
    print(f"Best epoch {best_ep} | Val loss {best:.4f} | Test loss {test_loss/test_total:.4f} Acc {test_correct/test_total:.4f}")

    torch.save({"trains_loss": trains_loss, "trains_accuracy": trains_accuracy,
                "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
                "best_epoch": best_ep,
                "test_loss": test_loss, "test_accuracy": test_correct/test_total},
                "train_results.pt")
    
    # loss and accuracy curve
    ep = range(1, EPOCH + 1)
    tl = trains_loss
    vl = validation_loss
    ta = trains_accuracy
    va = validation_accuracy

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(1, EPOCH)
    ax.plot(ep, tl, 'b-',  label='train loss')
    ax.plot(ep, vl, 'b--', label='val loss')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(ep, ta, 'r-',  label='train acc')
    ax2.plot(ep, va, 'r--', label='val acc')
    ax2.set_ylabel('accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.axvline(best_ep, ls=':', c='gray', label=f'best (ep {best_ep})')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='center right')
    plt.title(f'seed {SEED} | best epoch {best_ep} | test acc {test_correct/test_total:.3f}')
    plt.tight_layout()
    plt.savefig(f'curves_seed_{SEED}.png', dpi=150)