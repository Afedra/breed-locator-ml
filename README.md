# Breed Locator Machine Learning Module
This is a Python machine learning app that identifies breeds of animals from pictures.

## Environment setup
This project was developed and tested on Ubuntu 18.04 LTS.

### Install Python packages
This assumes you already have Python 3+ and pip installed.

You will need the following Python packages installed to work through this project:
* Scrapy (for scraping akc.org for a list of breed names and pics)
* mahotas (for calculating Haralick textures)
* imutils (image operations convenience functions)
* seaborn (for making nifty plots)
* pandas (for data operations)
* scikit-learn (machine learning library)
* progressbar (for showing eta and % completion of tasks)

These can be installed with the 'pip install -r requirements.txt' command.

### Install OpenCV 2.4.10
You will also need to install OpenCV.  If you are using an Anaconda distribution of Python, installing OpenCV is as easy as `conda install -c menpo opencv=2.4.11` in a terminal.

'''
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev gfortran libboost-all-dev python2.7-dev tcl-dev tk-dev python-tk python3-tk -y

wget -O opencv-2.4.10.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.10/opencv-2.4.10.zip/download
unzip opencv-2.4.10.zip
cd opencv-2.4.10
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON  -D BUILD_EXAMPLES=ON .. # you may have to tune these arguments if you're using a different OS than Ubuntu 16.04 LTS
make -j4 # change 4 to the number of cores in your CPU
sudo make install
'''

I have provided the install_cv2.sh file, which can be run in a Linux environent as `./install_cv2.sh` from a terminal.  This will install OpenCV 2.4.10, and has worked for me in Ubuntu 18.04 LTS.  To install on another OS, you will have to consult Google for now.

### Pre-processing the images
#### Overview
##### Preparing Images
* enables drawing rectangles around breeds in images -- WARNING: this task takes a long time to complete
* removes bounding boxes from accidental clicks
* uses OpenCV grabCut() to segment dog from background -- WARNING: this script takes hours to complete

##### Extracting Features
* extract 13-dim Haralick texture and color histogram from fore- and background of images -- WARNING: this script takes hours to complete
* perform PCA on Haralick, does some analysis on PCA and color histograms
* extract full 13x4-dim Haralick textures from the foreground of images -- WARNING: this script takes hours to complete
* extract color histograms from the foreground of images

#### Draw rectangles around animals
My first step was to go through images of breeds, outlining the breeds with a rectangle. 

The script will start logging the rectangles you draw as bounding boxes for breeds' bodies.  The controls for the interface are:
-right arrow = next available pic
-left arrow = previous available pic
-'n' = random pic
-'d' = next dog breed
-'b' = log coords of drawn rectangle around dog bodies
-'f' = log coords of drawn rectangle around dog faces
-'r' = reset all bounding boxes
-'q' = quit program

The bounding boxes for each image will be saved in the file 'pbreeds-bounding-boxes.pd.pk' in your pickle folder.

Finally, get rid of any tiny bounding boxes you may have accidentally created.  I accidentally created some with my laptop touchpad.

You can check the bounding boxes by running the file 'process_ims/check_bounding_boxes.py', which will draw the bounding boxes for bodies on the images, and then bounding boxes for heads, and can be advanced to the next image by pressing any key.

#### Grabcut dog foregrounds
The next step was to use the grabCut() function from OpenCV to remove the backgrounds from the images. Warning: this took about 3 hours on my machine.

#### Extract Haralick textures and color histograms
Next, extract our features from the images.  I chose to extract Haralick texture and color histograms from the foreground of the images. It will also display a plot comparing the foreground and background color histograms.  This script will take a long time, somewhere around 2 hours.

The Haralick features are calculated by breaking the forground up into small squares, calculating the Haralick feature on each square, and averaging them to arrive at one Haralick feature vector for the foreground.

Extract the 52-dimension Haralick features of only the foregrounds.
Extract the color histograms of only the foregrounds.
Extract the 20-dim PCA of the color histograms of the foregrounds.
    
#### Analyze Haralick textures
Extract the 13-dimension Haralick features and color histograms of the foreground and background.

### Machine learning
#### Overview
* machine_learning.py -- goes through machine learning algos and analysis
* check_robust.py -- checks performance on unseen data
* check_peturb.py -- checks reaction of model to noise in training and test data

#### Train and test the classifiers
Run 'process_ims/machine_learning.py' to see the performance of machine learning classifiers on the training data using different kinds of features.

#### Check robustness of model
Run 'process_ims/check_robust.py' and 'process_ims/check_peturb.py' to check the model's performance on unseen data and it's sensitivity to noise.
