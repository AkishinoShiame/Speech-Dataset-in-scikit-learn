import numpy as np
from PIL import Image
import pickle as pkl

train_path_Natural = "test/Natural/Natural_test-"
train_path_Negative = "test/Negative/Negative_test-"
train_path_Positive = "test/Positive/Positive_test-"

train_data = np.empty([52, 65536], dtype=int)
train_data_colord = np.empty([52, 196608], dtype=int)


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


def load_and_temp_data():
    for i in range(18):
        print("In Prograss part-1 " + str(i) + "/17")
        storage_data_gray(i, convert_file_gray(train_path_Positive + str(i) + ".png"))  # Positive
        storage_data_color(i, convert_file_color(train_path_Positive + str(i) + ".png"))  # Positive
    for i in range(14):
        print("In Prograss part-2 " + str(i) + "/13")
        storage_data_gray(i + 18, convert_file_gray(train_path_Natural + str(i) + ".png"))  # Natural
        storage_data_color(i + 18, convert_file_color(train_path_Positive + str(i) + ".png"))  # Natural
    for i in range(20):
        print("In Prograss part-3 " + str(i) + "/19")
        storage_data_gray(i + 32, convert_file_gray(train_path_Negative + str(i) + ".png"))  # Negative
        storage_data_color(i + 32, convert_file_color(train_path_Positive + str(i) + ".png"))  # Negative
    print(train_data)
    print(train_data_colord)
    print("Train data finished !")


def pkl_data():
    print("dumping data gray...")
    pkl.dump(train_data, open("test_data_gray.pkl", "wb"))
    print("Nice finished!")
    print("dumping data colord...")
    pkl.dump(train_data_colord, open("test_data_color.pkl", "wb"))
    print("Nice finished!")


if __name__ == "__main__":
    load_and_temp_data()
    pkl_data()
    print("All finished ! ")
