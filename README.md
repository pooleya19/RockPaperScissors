# Real-Time Interactive Rock-Paper-Scissors Using a Convolutional Neural Network
Developed by Adam Pooley, Oscar Jed Chuy, and Jack Curtis. \
Submitted as our final project for ECE 6254 Statistical Machine Learning at Georgia Tech. \
Video summary available at: [https://youtu.be/bNuyBQhNVYU](https://youtu.be/bNuyBQhNVYU)

## Setting Up the Environment
1. Clone this GitHub repository.
2. Install [Python](https://www.python.org/downloads/).
3. Install the required packages by running the following command in the cloned repository:
    ```
    py -m pip install -r requirements.txt
    ```

## Collecting a Dataset
1. Set up a fixed camera rig for training *and* testing. The picture below shows our fixed camera rig, using a USB webcam pointed at a sheet of poster board. \
    <img src="/README_Figures/CameraRig.jpg" alt="drawing" width="400"/>
3. Take pictures for training:
    1. In [DataCollection/DataCollector.py](/DataCollection/DataCollector.py), change line 10 to include the name of the existing folder you would like pictures saved in.
    2. Run the data collection script located in [DataCollection/](/DataCollection/) with the following command:
        ```
        py main_collection.py
        ```
    3. Once the script begins and the GUI appears, position your hand in the camera frame in a rock, paper, or scissors shape. Then, hold down 0, 1, or 2 respectively for rock, paper, or scissors, to take pictures at 30 Hz measuring 300x300 pixels. These pictures will be saved in the specified directory and labeled. Press ESCAPE or close the GUI to stop the script.
4. After taking pictures, flatten and save the pictures as ```.npy``` files:
    1. In [DataCollection/main_compaction.py](/DataCollection/main_compaction.py), change lines 10 and 11 to include the folder(s) containing input images and the folder in which the ```data.npy``` and ```labels.npy``` files will be saved.
    2. Run the data compaction script located in [DataCollection/](/DataCollection/) with the following command:
        ```
        py main_compaction.py
        ```
    3. If successful, you should be able to see new files in your specified output directory named ```data.npy``` and ```labels.npy```.
5. After flattening the images into ```data.npy``` and ```labels.npy``` files, post-process the data:
    1. In [DataCollection/main_postProcess.py](/DataCollection/main_postProcess.py), change line 4 to include the folder containing the unprocessed ```data.npy``` file.
    2. Run the post-processing script located in [DataCollection/](/DataCollection/) with the following command:
        ```
        py main_postProcess.py
        ```
    3. If successful, you should be able to see new files in your specified directory named ```data_gray.npy``` (which will be grayscaled at 300x300 pixels) and ```data_150_gray.npy``` (which will be grayscaled and downsampled to 150x150 pixels).

## Training the Model
1. Copy your ```data_150_gray.npy``` and ```labels.npy``` files into your desired folder in [Models/](/Models/).
2. In [Models/TrainModel.py](/Models/TrainModel.py), change lines 15-20 to reflect the data and label file paths, the output model ```.pkl``` file save path, whether to retrain the model or not (likely ```True```), the desired number of training epochs (we used ```200``` over the course of many hours), and whether to save the model (likely ```True```).
3. Run the (slow) model training script located in [Models/](/Models/) with the following command:
    ```
    py TrainModel.py
    ```
4. If successful, you should see a new ```.pkl``` file containing your trained model's parameters.

## Playing Rock-Paper-Scissors Against Your Model
1. In [Models/GUI_Gameplay/GUI_Gameplay.py](/Models/GUI_Gameplay/GUI_Gameplay.py), change line 24 to include the file path to the desired model parameter ```.pkl``` file.
2. Run the gameplay script located in [Models/GUI_Gameplay/](/Models/GUI_Gameplay/) with the following command:
    ```
    py GUI_Gameplay.py
    ```
3. If successful, the model should be loaded and a GUI should appear, showing the live camera feed.

### Gameplay GUI Instructions
- Press SPACE to play a round.
- AFTER a round has been played, press 0 (rock), 1 (paper), or 2 (scissors) to indicate the player's true move.
    - Used *only* for accuracy calculations.
- Press ESCAPE or close the window to stop the script. The model's true classification accuracy will be reported.

## Playing Rock-Paper-Scissors Against Your Model (While It Cheats)
1. In [Models/GUI_Cheater/GUI_Cheater.py](/Models/GUI_Cheater/GUI_Cheater.py), change line 23 to include the file path to the desired model parameter ```.pkl``` file.
2. Run the gameplay script located in [Models/GUI_Cheater/](/Models/GUI_Cheater/) with the following command:
    ```
    py GUI_Cheater.py
    ```
3. If successful, the model should be loaded and a GUI should appear, showing the live camera feed.
4. Similar to the actual gameplay script, press SPACE to play a round.
    - During gameplay, the model will classify your move in real-time and attempt to make its move to beat your current move (a strategy typically seen as cheating).
