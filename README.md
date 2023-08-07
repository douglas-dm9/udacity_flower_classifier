# AI Programming with Python Project - Data Scientist Nanodegree
### Flower-Species-Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

* Problem: Train an image classifier based on a per-trained neural network to recognize several species of flowers using the d [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  of 102 flower categories.

* Solution: The data was split into training, validation and test. In the training data some feature transformations as  random scaling, cropping, and flipping were applied to make the neural network generalize better. The validation set was used to monitoring the traning proccess, evaluating the classifier four times by epoch, but in a future work it can be used to optimize the neural network hyperparameters.

* Result: Using the test data, the model that was trained with 5 epochs and a lerning rate 0f 0.001 achieved a accuracy of 82%

#### Model Parameters


#### Running 

Execute the file train.py to train a new model from scratch

**Training**
* CLI: python train.py data_directory

##### Options:
* Directories where data is located and save directory: python train.py data_dir --save_dir save_directory
* Architecture: densenet121 or vgg16
* Hyperparameters: learning_rate, hidden_units, epochs
* Computing proccess unit: GPU or CPU

**Prediction**
Execute the file predict.py 

CLI: python predict.py /path/to/image checkpoint
##### Options:
* Top K classes: --top_k
* Mapping file from flower real names: --category_names cat_to_name.json
* Computing proccess unit: GPU or CPU

**Example of CLI**
* Training: python train.py ./flowers  --epochs 5 --learning_rate 0.001
* Prediction: python predict.py ./flowers/test/10/image_07090.jpg checkpoint.pth --top_k 3
