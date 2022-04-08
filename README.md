# CV-assignment-1
## Training Process:\
To train this model,we first need to initialize the Neural Network class to build to a two-layers network.
We can assign the Learning rate, dimension of inputs, dimension of hidden layer, dimension of outputs and regulation intensity to the network.
After that, we should use the train function of the Neural Network class to train the model.Here, we need to put the images and the labels of the images into the function.What's more, we can choose the iter numbers for the training.

## Testing Process:\
To test this model,we can simply use the get_accuracy function of the neural network instance that we trained before. After we put the test images and their labels into that function, the function will return the accuracy of the model's prediction.

## Save and load\
To save this model, we can use the save_model function of the Neural Network class.This function will save four npy files under the same directory as the python file.
The four files indicate the parameter matrix of the first layer, the parameter matrix of the second layer, the bias of the first layer and the bias of the second layer.\

To load this model, we can first use the load_np function of the class to get trained parameters.This function will return the parameter matrix of the first layer, the parameter matrix of the second layer, the bias of the first layer and the bias of the second layer.Note that in order to run this function correctly, we need to put the four npy files under the same directory as the python file.And the four npy files should be named as "w1.npy", "w2.npy","b1.npy","b2.npy" corresponded to the parameters it stored.Then we can use the load model function to assign the parameters to the model.
