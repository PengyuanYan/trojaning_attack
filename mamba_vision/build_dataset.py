from pathlib import Path
import os
from PIL import Image
import csv
from collections import Counter
import io
import torch

if __name__ == "__main__":
    MAMBA = Path(__file__).resolve().parent
    ROOTS = [MAMBA.parent / "vggface2/data/test",
             MAMBA.parent / "vggface2/data/train"]

    identy_folders = {}
    for root in ROOTS:
        for folder in os.listdir(root):
            folder_path = root / folder
            if folder_path.is_dir():
                identy_folders[folder] = folder_path

    # in default sort keys and retrun the results
    chosen_class_ids = sorted(identy_folders)[:100]

    BBROOTS = [MAMBA.parent / "vggface2/meta/bb_landmark/loose_bb_test.csv",
               MAMBA.parent / "vggface2/meta/bb_landmark/loose_bb_train.csv"]

    bounding_box = {}
    for excel in BBROOTS:
        with open(excel) as f:
            for row in csv.DictReader(f):
                full_name_id = row["NAME_ID"]
                class_id = full_name_id.split("/")[0]
                if class_id in chosen_class_ids:
                    bounding_box[full_name_id] = (float(row["X"]), float(row["Y"]),
                                                  float(row["W"]), float(row["H"]))
    print(len(bounding_box))

    images = []
    labels = []
    missing = 0
    padded = 0
    worst = 0.0
    shrunk = 0
    clamped = 0
    per_class_clamped = Counter()
    worst_shift = 0.0

    for label, id in enumerate(chosen_class_ids):
        for file_name in sorted(os.listdir(identy_folders[id])):
            full_name_id = f"{id}/{file_name.split('.')[0]}"
            if full_name_id not in bounding_box:
                print("X")
                missing += 1
                continue

            image = Image.open(identy_folders[id] / file_name).convert("RGB")
            W, H = image.size
            x, y, w, h = bounding_box[full_name_id]
            centre_x = x + w/2
            centre_y = y + h/2
            side = max(w, h) * 1.1
            
            if side > min(W, H):
                side = float(min(W, H))
                shrunk += 1

            want_x = centre_x - side/2
            want_y = centre_y - side/2
            x0 = min(max(0.0, want_x), W - side)
            y0 = min(max(0.0, want_y), H - side)

            if x0 != want_x or y0 != want_y:
                clamped += 1
                per_class_clamped[label] += 1
                worst_shift = max(worst_shift, max(abs(x0-want_x),abs(y0-want_y))/side)
            
            cropped_image = image.crop((x0,y0,x0+side,y0+side)).resize((256,256), Image.BICUBIC)

            buffer = io.BytesIO()
            cropped_image.save(buffer, format="JPEG", quality=92)
            images.append(buffer.getvalue())
            labels.append(label)

    print(f"images:{len(images)}")
    print(f"skipped:{missing}")
    print(f"clamped:{clamped}({100*clamped/max(len(images),1):.1f}%)")
    print(f"most clamped class: {[(chosen_class_ids[c], k) for c, k in per_class_clamped.most_common(5)]}")
    print(f"shrunk:{shrunk}")
    print(f"worst shift:{100*worst_shift:.1f}%")
    torch.save((images, labels), "subset100.pt")

