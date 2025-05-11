import PIL
import numpy as np
import matplotlib.pyplot as plt
import os, random


def main():
    path = "../DATA/clouds/clouds_train/"
    folders = [r"cirriform clouds", r"clear sky", r"cumulonimbus clouds",
               r"cumulus clouds", r"high cumuliform clouds",
               r"stratiform clouds", r"stratocumulus clouds"]
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes = axes.flatten()

    for i in range(6):
        i_rand = np.random.randint(0,7,1)[0]
        curr_path = path + folders[i_rand] + "/" + random.choice(os.listdir(path + folders[i_rand]))
        image = PIL.Image.open(curr_path)
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(folders[i_rand])

    plt.suptitle("The Clouds Dataset")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()