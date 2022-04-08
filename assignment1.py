"""
Author: FengZhongrui 18307110014
Filename: assignment 1
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image


class LearningRate:
    def __init__(self, initializer=0.1, gamma=0.6):
        self.rate = initializer
        self.gamma = gamma

    def descend(self):
        self.rate *= self.gamma


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, LAM):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.w1 = np.random.normal(size=(hidden_nodes, input_nodes))/input_nodes
        self.w2 = np.random.normal(size=(output_nodes, hidden_nodes))/hidden_nodes
        self.b1 = np.random.normal(size=(hidden_nodes,))
        self.b2 = np.random.normal(size=(output_nodes,))
        self.hidden_grad = np.zeros(hidden_nodes)
        self.learning_rate = learning_rate
        self.accuracy = []
        self.loss = []
        self.image_array = []
        self.lam =LAM
        self.test_accuracy = []
        self.test_loss = []

    def load_model(self, w1, w2, b1, b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def load_np(self):
        """

        :return: w1, w2, b1, b2
        """
        return np.load("w1.npy"), np.load("w2.npy"), np.load("b1.npy"), np.load("b2.npy")

    def save_model(self):
        np.save("w1.npy",self.w1)
        np.save("w2.npy", self.w2)
        np.save("b1.npy", self.b1)
        np.save("b2.npy", self.b2)

    def relu_activation(self, inputs):
        return np.maximum(inputs, 0)

    def get_loss(self, true, prediction):
        return -true*np.log(prediction) + self.lam/2*(np.linalg.norm(self.w2,ord='fro')**2+np.linalg.norm(self.w1,ord='fro')**2)


    def first_layer(self, inputs):
        return np.dot(self.w1, inputs) + self.b1

    def hidden_layer(self, inputs):
        return np.dot(self.w2, inputs) + self.b2

    def sigmoid_layer(self, input):
        return 1/(1+np.exp(-input))

    def softmax_layer(self, input):
        exp = np.exp(input)
        return exp/sum(exp)

    def derive_softmax(self,f):
        return f - np.square(f)

    def derive_sigmoid(self, output):
        return output*(1-output)

    def update_first_layer(self,input_value,hidden_value):
        rate = self.learning_rate.rate
        grad = self.hidden_grad*self.derive_sigmoid(hidden_value)
        for i in range(self.hidden_nodes):
            self.w1[i] -= rate*(grad[i]*input_value + self.lam*self.w1[i])
            self.b1[i] -= rate*grad[i]

    def update_second_layer(self, softmax_pre, real_value, hidden_value):
        rate = self.learning_rate.rate
        # gradient_y = -real_value/softmax_pre
        # gradient_y = self.derive_softmax(softmax_pre)*gradient_y
        gradient_y = softmax_pre - real_value
        self.hidden_grad = np.dot(gradient_y.T, self.w2)
        for i in range(self.output_nodes):
            self.w2[i] -= rate*(hidden_value*gradient_y[i] + self.lam*self.w2[i])
            self.b2[i] -= rate*gradient_y[i]
        # for i in range(self.hidden_nodes):
        #     self.hidden_grad[i] += np.sum(self.w2[:, i]*gradient_y)

    def forward(self, input):
        first_layer_output = self.first_layer(input)
        hidden_output = self.sigmoid_layer(first_layer_output)
        prediction = self.hidden_layer(hidden_output)
        softmax_prediction = self.softmax_layer(prediction)

        return softmax_prediction, hidden_output

    def train_model(self, inputs, labels, test_inputs=None, test_labels=None, max_iters=100000):
        iter = 0
        prediction_list = []
        pre_test_list = []
        loss_list = []
        test_loss_list = []
        while iter < max_iters:
            i = iter%(len(inputs))
            input_normal = np.array(inputs[i], dtype=np.float32)/255
            softmax_prediction, hidden_output = self.forward(input_normal)
            loss = self.get_loss(labels[i], softmax_prediction)
            loss_list.append(loss)
            pre_label = self.get_category(softmax_prediction)
            prediction_list.append(pre_label == np.argmax(labels[i]))

            if test_inputs is not None:
                j = iter % (len(test_inputs))
                test_normal = np.array(test_inputs[j], dtype=np.float32)/255
                test_prediction, _ = self.forward(test_normal)
                test_loss = self.get_loss(test_labels[j], test_prediction)
                test_loss_list.append(test_loss)
                test_pre_label = self.get_category(test_prediction)
                pre_test_list.append(test_pre_label == np.argmax(test_labels[j]))


            self.update_second_layer(softmax_prediction, labels[i], hidden_output)
            self.update_first_layer(input_normal, hidden_output)
            if len(prediction_list) == 1000:
                self.accuracy.append(sum(prediction_list)/1000)
                self.loss.append(np.mean(loss_list))
                loss_list = []
                prediction_list = []

                if test_inputs is not None:
                    self.test_accuracy.append(sum(pre_test_list) / 1000)
                    self.test_loss.append(np.mean(test_loss_list))
                    pre_test_list = []
                    test_loss_list = []

                _ = self.get_para_pca()
                if iter > 0.2*max_iters:
                    self.learning_rate.descend()


            iter += 1

    def get_category(self, prediction):
        return np.argmax(prediction)

    def get_para_pca(self):
        """

        :return: Matrix 784 * 3
        """
        para = np.dot(self.w1.T, self.w2.T)
        para -= np.mean(para)
        cov_para = np.dot(para.T, para)/(para.shape[0]-1)
        U, S, V = np.linalg.svd(cov_para)
        df_reduced = np.dot(para, U[:, :3])
        df_min = np.min(df_reduced)
        df_reduced -= df_min
        df_max = np.max(df_reduced)
        df_reduced /= df_max
        df_reduced *= 255
        df_reduced //= 1
        self.image_array.append(df_reduced.reshape(28,28,3))
        return df_reduced

    def get_accuracy(self, inputs, labels):
        input_len = len(inputs)
        correct_num = 0
        for i in range(input_len):
            input_normal = np.array(inputs[i], dtype=np.float32) / 255
            first_layer_output = self.first_layer(input_normal)
            hidden_output = self.sigmoid_layer(first_layer_output)
            prediction = self.hidden_layer(hidden_output)
            prediction = self.softmax_layer(prediction)
            pre_label = self.get_category(prediction)
            if pre_label == labels[i]:
                correct_num += 1

        return correct_num/input_len

    def plot_accuracy(self):
        plt.plot(self.accuracy,c='#2ca02c',label='train_set')
        plt.plot(self.test_accuracy, c='#ff7f0e', label='test_set')
        plt.legend()
        plt.title("Accuracy to every 1000 iters")
        plt.xlabel("iters(1000)")
        plt.ylabel("accuracy")
        plt.show()

    def generate_vedio(self):
        """
        Need to create a empty folding called para in order to store the images
        :return: Generate a gif of how parameters change during training
        """
        gif_list = []
        for i in range(len(self.image_array)):
            imageio.imwrite(f'para/{i}.jpg',self.image_array[i])
            gif_list.append(imageio.imread(f'para/{i}.jpg'))
        imageio.mimsave('para.gif', gif_list, fps=5)

    def plot_loss(self):
        plt.plot(self.loss, c='#2ca02c',label='train-set')
        plt.plot(self.test_loss, c='#ff7f0e', label='test_set')
        plt.legend()
        plt.title("Loss to every 1000 iters")
        plt.xlabel("iters(1000)")
        plt.ylabel("loss")
        plt.show()




def search_para(train_image, train_label_one_hot, train_label, test_image, test_label_one_hot, test_label):
    """
    find the best hyper parameters
    :return: dimension of hidden layer, initial_learning_rate, regularization intensity
    """
    Hidden_nodes = list(range(20, 200, 10))
    hidden_search_result = []
    LAM = 0.0001
    for i in Hidden_nodes:
        learning_rate = LearningRate(0.1)
        two_layer_network = NeuralNetwork(28 * 28, i, 10, learning_rate, LAM)
        two_layer_network.train_model(train_image, train_label_one_hot, test_image, test_label_one_hot)
        accuracy = two_layer_network.get_accuracy(test_image, test_label)
        hidden_search_result.append(accuracy)
        print(f"accuracy={accuracy},hidden_nodes={i}")
    # find that when hidden nodes = 50, the accurarcy is best

    best_hidden = Hidden_nodes[hidden_search_result.index(max(hidden_search_result))]
    print("best_hidden=", best_hidden)

    learning_rate_choices = np.linspace(0.05, 0.2, 16)
    lr_search_result = []
    for j in learning_rate_choices:
        learning_rate = LearningRate(j)
        two_layer_network = NeuralNetwork(28 * 28, best_hidden, 10, learning_rate, LAM)
        two_layer_network.train_model(train_image, train_label_one_hot, test_image, test_label_one_hot)
        accuracy = two_layer_network.get_accuracy(test_image, test_label)
        lr_search_result.append(accuracy)
        print(f"accuracy={accuracy}, Learning_rate = {j}")

    best_lr = learning_rate_choices[lr_search_result.index(max(lr_search_result))]
    print("best_lr=", best_lr)
    # find lr=0.2

    lam_choices = np.linspace(0.0001, 0.00001, 10)
    lam_search_results = []
    for k in lam_choices:
        LAM = k
        learning_rate = LearningRate(best_lr)
        two_layer_network = NeuralNetwork(28 * 28, best_hidden, 10, learning_rate, LAM)
        two_layer_network.train_model(train_image, train_label_one_hot, test_image, test_label_one_hot)
        accuracy = two_layer_network.get_accuracy(test_image, test_label)
        lam_search_results.append(accuracy)
        print(f"accuracy={accuracy},LAM = {k}")

    best_lam = lam_choices[lam_search_results.index(max(lam_search_results))]
    print("best_lam=", best_lam)
    # find lam = 8e-05
    return best_hidden, best_lr, best_lam


def train(train_image, train_label_one_hot, test_image, test_label_one_hot, lam = 8e-05, lr=0.2, hidden=50, iter=100000):
    """

    :param train_image: images of train set
    :param train_label_one_hot: labels of train set with one hot coding
    :param test_image: images of test set
    :param test_label_one_hot: labels of test set with one hot coding
    :param lam: regularization intensity
    :param lr:  learning rate
    :param hidden: dimension of hidden layer
    :return:
    """
    LAM = lam
    learning_rate = LearningRate(lr)
    best_hidden = hidden
    two_layer_network = NeuralNetwork(28 * 28, best_hidden, 10, learning_rate, LAM)
    two_layer_network.train_model(train_image, train_label_one_hot, test_image, test_label_one_hot,max_iters=iter)
    two_layer_network.plot_accuracy()
    two_layer_network.plot_loss()
    return two_layer_network


if __name__ == '__main__':
    np.random.seed(42)
    """
    get [train_image] [get train_label] [get test_image] [get test_label]
    """
    with open("Minist/train-images.idx3-ubyte", 'rb') as f:
        file = f.read()
        image = [int(str(item), 16) for item in file[16:]]
        train_image = np.array(image, dtype=np.int8).reshape(-1, 28 * 28)

    with open("Minist/train-labels.idx1-ubyte", 'rb') as f:
        file = f.read()
        label = [int(str(item), 16) for item in file[8:]]
        train_label = np.array(label, dtype=np.uint8)

        num_class = 10
        train_label_one_hot = np.eye(num_class)[train_label]

    with open("Minist/t10k-images.idx3-ubyte", 'rb') as f:
        file = f.read()
        image = [int(str(item), 16) for item in file[16:]]
        test_image = np.array(image, dtype=np.uint8).reshape(-1, 28 * 28)

    with open("Minist/t10k-labels.idx1-ubyte", 'rb') as f:
        file = f.read()
        label = [int(str(item), 16) for item in file[8:]]
        test_label = np.array(label, dtype=np.uint8)

        num_class = 10
        test_label_one_hot = np.eye(num_class)[test_label]

    # run best model
    # best learning rate = 0.2
    # best regulariation intensity = 8e-05
    # best hidden layer dimension = 50
    # search_para()   function used to search hpyer parameters

    two_layer_network = train(train_image, train_label_one_hot, test_image, test_label_one_hot)
    accuracy = two_layer_network.get_accuracy(test_image, test_label)
    # A function used to show how parameters changes
    # two_layer_network.generate_vedio()
    print("accuracy=", accuracy)
    # The accuracy on test set is 0.9106