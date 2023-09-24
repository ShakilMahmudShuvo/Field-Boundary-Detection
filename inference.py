import numpy as np
import os
import glob
import keras
import rasterio as rio
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize image to give a meaningful output."""
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


# Combo Loss - Dice + Binary crossentropy

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice

def bce_dice_loss(y, p):
    return dice_coef_loss_bce(y, p, dice=0.9, bce=0.1)


#https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall_   = recall(y_true, y_pred)
    return 2*((precision*recall_)/(precision+recall_+K.epsilon()))

new_dir = "D:/New folder/Spacenus_ML_Engineer_Assignment/Field_boundary_detection_test"


train_source_items = f"{new_dir}/train"
train_label_items = f"{new_dir}/labels/train"

#image snapshot dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 4 #we have the rgba bands
BATCH_SIZE = 4

# You need to specify the path to your train data folder
train_data = f"{new_dir}/train_data"

# Get all the subdirectories from the train data folder
subdirs_1 = next(os.walk(train_data))[1] #Get all subdirectories, e.g. 2021_01, 2021_02, etc.




# Specify the path to the private test imagery folder
private_test_items = f"{new_dir}/private"

# Initialize a variable to store the total count
total_tif_count = 0

# Use a recursive search to find all .tif files in the directory and its subdirectories
for root, dirs, files in os.walk(private_test_items):
    for file in files:
        if file.endswith(".tif"):
            total_tif_count += 1

# Print the total count of .tif files
print(f"Total .tif files in {private_test_items}: {total_tif_count}")


# Specify the path to the private test imagery folder
private_test_items = f"{new_dir}/private"

# Get all the subdirectories representing years and months
subdirs = next(os.walk(private_test_items))[1]
print(len(subdirs))
# Initialize empty arrays for X_test and test_tile_ids
# Initialize empty arrays for X_test and test_tile_ids
X_test = np.empty((72, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*len(subdirs_1)), dtype=np.float32)
test_tile_ids = []

i = 0
idx = 0  # Move idx outside the inner loop
for subdir in subdirs:
    test_id_dirs = next(os.walk(f"{private_test_items}/{subdir}"))[1]

    for test_id_dir in test_id_dirs:
        bd1 = rio.open(
            f"{private_test_items}/{subdir}/{test_id_dir}/{test_id_dir}_B01.tif"
        )
        bd1_array = bd1.read(1)
        bd2 = rio.open(
            f"{private_test_items}/{subdir}/{test_id_dir}/{test_id_dir}_B02.tif"
        )
        bd2_array = bd2.read(1)
        bd3 = rio.open(
            f"{private_test_items}/{subdir}/{test_id_dir}/{test_id_dir}_B03.tif"
        )
        bd3_array = bd3.read(1)
        bd4 = rio.open(
            f"{private_test_items}/{subdir}/{test_id_dir}/{test_id_dir}_B04.tif"
        )
        bd4_array = bd4.read(1)

        field = np.dstack((bd4_array, bd3_array, bd2_array, bd1_array))
        field = np.sqrt(field)
        # print(field)
        # data standardization
        for c in range(field.shape[2]):
            mean = field[:, :, c].mean()
            std = field[:, :, c].std()
            field[:, :, c] = (field[:, :, c] - mean) / std
        X_test[idx][:, :, :IMG_CHANNELS] = field
        idx += IMG_CHANNELS  # Increment idx by IMG_CHANNELS for each test_id_dir
        test_tile_ids.append(test_id_dir)
    i += 1  # Increment i by 1 for each subdir

# load best model to run predictions
# You need to specify the path to your best model file
best_model_path = f'D:/New folder/Saved Models/final.h5'

sm.set_framework('tf.keras')
sm.framework()
# Load your best model
model = keras.models.load_model(
    best_model_path,
    custom_objects={"bce_dice_loss": bce_dice_loss, "f1": f1, "recall": recall},
)


model.load_weights("D:/New folder/Saved Models/best_model.h5")

preds = model.predict(X_test, verbose=1, batch_size=BATCH_SIZE)



#-------- Saving the Inference---------
# Convert predictions into binary masks
preds = (preds > 0.5).astype(np.uint8)

# Create a folder for inference
os.makedirs(f"{new_dir}/inference", exist_ok=True)

# Loop over the subdirectories and test ids
idx = 0
for subdir in subdirs:
    test_id_dirs = next(os.walk(f"{private_test_items}/{subdir}"))[1]

    for test_id_dir in test_id_dirs:
        # Create a subfolder for each test id
        os.makedirs(f"{new_dir}/inference/{subdir}/{test_id_dir}", exist_ok=True)

        # Open the original image
        bd1 = rio.open(
            f"{private_test_items}/{subdir}/{test_id_dir}/{test_id_dir}_B01.tif"
        )

        # Open a new file for writing the mask image
        with rio.open(
            f"{new_dir}/inference/{subdir}/{test_id_dir}/{test_id_dir}.tif",
            "w",
            **bd1.profile,
        ) as dst:
            # Write the mask image to the file
            dst.write(preds[idx][:, :, 0], 1)

        # Increment idx by IMG_CHANNELS for each test id
        idx += IMG_CHANNELS


