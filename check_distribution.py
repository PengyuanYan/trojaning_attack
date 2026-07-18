import os, collections
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

if __name__ == "__main__":
    folder = 'filtered_fc6_81_694_1_1_0.3_vgg_mfv_true'
    counts = collections.Counter(
        int(f[:-4].split('_')[2]) for f in os.listdir(folder) if f.endswith('.jpg')
    )

    classes = range(max(counts) + 1)
    images = [counts.get(i, 0) for i in classes]
    peak = max(counts, key=counts.get)

    (peak_height, width), _ = curve_fit(lambda distance, peak_height, decay_scale: peak_height * np.exp(-distance / decay_scale),
                              np.abs(np.array(classes) - peak), images, p0=[30, 40])

    print('width:', width)

    plt.figure(figsize=(14, 5))
    plt.bar(classes, images, width=1.0)
    plt.xlabel('class number')
    plt.ylabel('number of images')
    plt.title('Paper dataset: images per class')
    plt.tight_layout()
    plt.savefig('paper_dataset_distribution.png', dpi=130)