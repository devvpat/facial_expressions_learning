import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

DATA_CSV_PATH = Path("./data/legend.csv")
DATA_CSV_IMAGE_HEADER = "image"
DATA_CSV_CLASS_HEADER = "emotion"
DATA_IMAGES_PATH = Path("./images")
DATA_IMG_SIZE = (350, 350)
DATA_IMG_RESIZE = (100, 100)    # change this to generate new txt file of specified image sizes

def convert_images(resize_shape: Tuple[int, int], return_data_early: bool) -> Optional[Tuple[List[List[int]], List[str]]]:
    df = pd.read_csv(DATA_CSV_PATH)
    data_X = []
    data_y = []

    print("--- Done reading legend.csv ---")

    for index, row in df.iterrows():
        img_path = DATA_IMAGES_PATH / row[DATA_CSV_IMAGE_HEADER]       # create image path
        img = Image.open(img_path).convert('L')     # open image in grayscale

        img = img.resize(resize_shape)
        img_vec = np.array(img).flatten()   # all img grayscale pixels in a 1d array
        data_X.append(img_vec)
        data_y.append(row[DATA_CSV_CLASS_HEADER])   # separate since copying 350*350 array is expensive
        print(f"{index / df.shape[0]:.2f}%")

    print("--- Done turning images into vectors ---")

    if return_data_early:
        return data_X, data_y

    output_file = Path(f"./data_images_{DATA_IMG_RESIZE[0]}_{DATA_IMG_RESIZE[1]}.txt") 

    with open(output_file, "w") as file:
        for ind, (data, classification) in enumerate(zip(data_X, data_y)):
            print(f"{ind / len(data_X):.2f}%")
            for val in data:
                file.write(f"{val} ")
            file.write(f"{classification}\n")
            
    print(f"--- Done converting images to txt at: {output_file} ---")

    # data = np.genfromtxt(DATA_OUTPUT_FILE)   


if __name__ == "__main__":
    convert_images(DATA_IMG_RESIZE, False)
