import numpy as np
from PIL import Image
import pickle as pkl

train_path_Natural = "train/Natural/Nature_train-"
train_path_Negative = "train/Negative/Negative_train-"
train_path_Positive = "train/Positive/Positive_train-"

train_data = np.empty([600, 65536], dtype=int)
train_label = np.empty([600], dtype=int)

train_data_colord = np.empty([600, 196608], dtype=int)


def convert_file_gray(filename):
    img_gray = Image.open(filename).convert('L')
    resized_gray = img_gray.resize((256, 256))
    gray_array = np.array(resized_gray)
    flatten_gray = gray_array.flatten()
    return flatten_gray

def convert_file_color(filename):
    img = Image.open(train_path_Natural+'1.png').convert('RGB')
    resized_img = img.resize((256, 256))
    img_array = np.array(resized_img)
    flatten_img = img_array.flatten()
    return flatten_img


def storage_data_gray(index, data):
    train_data[index] = data


def storage_data_color(index, data):
    train_data_colord[index] = data


def label_data():
    print("Labeling data")
    for i in range(600):
        if i >= 400:
            train_label[i] = 2  # Positive
        elif i >=200:
            train_label[i] = 1  # Natural
        else:
            train_label[i] = 0  # Negative
    print(train_label)
    print("Finished labeling...")


def load_and_temp_data():
    for i in range(600):
        print("In Prograss train " + str(i) + "/600")
        if i >= 400:
            storage_data_gray(i, convert_file_gray(train_path_Positive + str(i - 400) + ".png"))  # Positive
            storage_data_color(i, convert_file_color(train_path_Positive + str(i - 400) + ".png"))  # Positive
        elif i >=200:
            storage_data_gray(i, convert_file_gray(train_path_Natural + str(i - 200) + ".png"))  # Natural
            storage_data_color(i, convert_file_color(train_path_Positive + str(i - 200) + ".png"))  # Natural
        else:
            storage_data_gray(i, convert_file_gray(train_path_Negative + str(i) + ".png"))  # Negative
            storage_data_color(i, convert_file_color(train_path_Positive + str(i) + ".png"))  # Negative
    print(train_data)
    print(train_data_colord)
    print("Train data finished !")


def pkl_data():
    print("dumping data gray...")
    pkl.dump((train_data, train_label), open("train_data_gray.pkl", "wb"))
    print("Nice finished!")
    print("dumping data colord...")
    pkl.dump((train_data_colord, train_label), open("train_data_color.pkl", "wb"))
    print("Nice finished!")


if __name__ == "__main__":
    load_and_temp_data()
    label_data()
    pkl_data()
    print("All finished ! ")
