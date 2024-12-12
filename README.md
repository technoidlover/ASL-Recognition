# ASL(american sign language) recognition

## Introduction
This project is designed to recognize gestures from video files or camera using a trained deep learning model. The project includes scripts for training the model, processing video files, and saving the results or recognize lifetime by using camera.


## How to Train the Model

1. **Prepare the Dataset**: Demo data I get from [MiAI](https://miai.vn/2019/09/30/xay-dung-he-thong-nhan-dien-thu-ngu-ngon-ngu-ky-hieu-tay-de-giao-tiep-voi-nguoi-khuyet-tat/), just keep using it with 5 char. Demo data image for train in `data` folder
2. **Train the Model**: Run the training script to train the model. You can use the `train_model.py` script in the `reg` directory.

    ```sh
    python train_model.py
    ```

    This script will load the data, create the model, and train it. The trained model will be saved in the `models` directory.

## How to Use the Model
you can use my pre-trained model in model folder, or if you want customize model and re-train, just delete my model or chang file name (file .keras and .h5)

1. **Process Video Files**: Use the [dec_mp4.py](http://_vscodecontentref_/7) script in the [reg](http://_vscodecontentref_/8) directory to process video files and recognize gestures.

    ```sh
    python dec_mp4.py
    ```

    This script will open a video file, process each frame, and display the results. The recognized gestures and timestamps will be saved in a text file.

2. **Save Results**: The [save_results](http://_vscodecontentref_/9) function in the [dec_mp4.py](http://_vscodecontentref_/10) script allows you to save the recognition results to a text file.
3. **Using with camera**: Run `detection_temp.py` or `dec_cam.py` file
    ```sh
    python dec_cam.py
    ```
## Dependencies
- Python 3.12
- OpenCV
- Keras
- TensorFlow
- NumPy
- Scikit-learn
- Pillow
# Install the dependencies using pip:
```bash
pip install -r requirements.t
```
## Update kaggle dataset
- I was updated training file using [kaggle dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset), just download and add data to /data folder
- /data folder have structure:
```
/data
├── 1
│   ├── 1_0_0.jpg
│   ├── 1_0_1.jpg
│   └── ...
├── 2
│   ├── 2_0_0.jpg
│   ├── 2_0_1.jpg
│   └── ...
└── ...
```
