# GTSRB Traffic Sign Classifier (CNN → TFLite)

A small CNN that recognises 43 types of German traffic signs, trained on Kaggle and converted to a ~157 KB TensorFlow Lite model so it can run on embedded devices.

| | |
|---|---|
| **Test accuracy** | **98.0%** (12,630 unseen images) |
| **Validation accuracy** | 99.9% (see the note below on why this is higher) |
| **Model size** | ~146k parameters, 157 KB as `.tflite` |
| **Input** | 64×64 RGB image |

## Dataset

[GTSRB – German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (Kaggle). Original benchmark page: <https://benchmark.ini.rub.de/gtsrb_dataset.html>

- 39,209 training images and 12,630 test images, listed in `Train.csv` and `Test.csv`
- Each row has the image path and a `ClassId` (0–42), which is our label

### The 43 classes we predict

| ID | Sign | ID | Sign |
|---|---|---|---|
| 0 | Speed limit 20 km/h | 22 | Bumpy road |
| 1 | Speed limit 30 km/h | 23 | Slippery road |
| 2 | Speed limit 50 km/h | 24 | Road narrows on the right |
| 3 | Speed limit 60 km/h | 25 | Road work |
| 4 | Speed limit 70 km/h | 26 | Traffic signals |
| 5 | Speed limit 80 km/h | 27 | Pedestrians |
| 6 | End of speed limit 80 km/h | 28 | Children crossing |
| 7 | Speed limit 100 km/h | 29 | Bicycles crossing |
| 8 | Speed limit 120 km/h | 30 | Beware of ice/snow |
| 9 | No passing | 31 | Wild animals crossing |
| 10 | No passing for vehicles over 3.5 t | 32 | End of all speed and passing limits |
| 11 | Right-of-way at next intersection | 33 | Turn right ahead |
| 12 | Priority road | 34 | Turn left ahead |
| 13 | Yield | 35 | Ahead only |
| 14 | Stop | 36 | Go straight or right |
| 15 | No vehicles | 37 | Go straight or left |
| 16 | Vehicles over 3.5 t prohibited | 38 | Keep right |
| 17 | No entry | 39 | Keep left |
| 18 | General caution | 40 | Roundabout mandatory |
| 19 | Dangerous curve to the left | 41 | End of no passing |
| 20 | Dangerous curve to the right | 42 | End of no passing by vehicles over 3.5 t |
| 21 | Double curve | | |

## Some Notes - 
**Data**
- **Shuffle before splitting.** `Train.csv` is sorted by class. Without shuffling, the validation split (the last 25% of rows) contained only a few classes the model had never trained on, so validation accuracy sat at about 6–10% while training accuracy was 99%. 
- **75 / 25 train–validation split** (29,407 / 9,802 images)
- **64×64 images.** Signs are small and simple. 64×64 keeps the shape and symbols readable while staying cheap
- **VGG16 `preprocess_input`.** Converts RGB to BGR and subtracts the ImageNet mean, which centres the pixel values and helps training converge. Note: whatever runs the TFLite model must apply the same preprocessing to its input.
- **Batch size 64** and **`seed=42`** for a good speed/stability balance and reproducible results, chosen arbitrarily.

**Model** (3 blocks of convolutions, 32 -> 64 -> 128 filters)

- **Augmentation layers**: small random rotation (+-5%), zoom (10%) and shift (10%). Real photos of signs are never perfectly centred. **No flipping**, because flipping would turn "turn left" into "turn right".
- **Doubling the filters** each block: early layers pick up edges and colours, later layers combine them into shapes and symbols. Pooling halves the image between blocks (64 -> 32 -> 16 -> 8).
- **BatchNormalization** makes training faster and more stable.
- **Dropout 0.25** randomly switches off units during training to reduce overfitting.
- **GlobalAveragePooling** instead of `Flatten` + a big `Dense` layer. It cuts the parameter count massively.
- **Softmax with 43 outputs**: one probability per sign.

**Training**

- **Adam, learning rate 0.001**
- **`sparse_categorical_crossentropy`**
- **Up to 30 epochs, with `EarlyStopping`** (patience 5, restores the best weights) 
- **`ReduceLROnPlateau`** (halve the learning rate after 2 epochs without improvement, floor 1e-6)

## Results

- Training accuracy reached ~99.7% and validation accuracy ~99.9%.
- **Test accuracy: 98.0%.** - Validation is higher because GTSRB images come in sequences of about 30 photos of the same physical sign, so a random split puts near-identical frames in both training and validation.

## TFLite conversion

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quantization
tflite_model = converter.convert()
```

`Optimize.DEFAULT` stores the weights as 8-bit integers, making the file roughly 4x smaller (about **157 KB**) with no calibration data needed. The input stays as float, so remember the preprocessing step above.

## Running it

1. Open the notebook on Kaggle and add the dataset linked above (GPU optional but faster) or download the dataset locally and attach the path in the `train_path`
2. Run the cells, and save the models.