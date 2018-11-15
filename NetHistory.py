import keras
import matplotlib.pyplot as plt

class NetHistory(keras.callbacks.Callback):
    def __init__(self):
        self.clear()
    
    def on_train_begin(self, logs={}):
        pass
        
    
    def on_epoch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.test_acc.append(logs.get('val_acc'))
        self.train_loss.append(logs.get('loss'))
        self.test_loss.append(logs.get('val_loss'))
        
    def on_train_end(self, logs=None):
        pass
    
    def showAccuracy(self, figsize=(8,8)):
        plt.figure(figsize=figsize)
        plt.plot(self.train_acc, label='training accuracy')
        plt.plot(self.test_acc, label='testing accuracy')
        plt.legend()
        plt.show()
    def showLoss(self, figsize=(8,8)):
        plt.figure(figsize=figsize)
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.test_loss, label='testing loss')
        plt.legend()
        plt.show()
    
    def show(self, figsize=(8,8)):
        fig = plt.figure(figsize=figsize)
        fig.add_subplot(2,1,1)
        plt.plot(self.train_acc, label='training accuracy')
        plt.plot(self.test_acc, label='testing accuracy')
        plt.legend()
        plt.show()
        fig.add_subplot(2,1,2)
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.test_loss, label='testing loss')
        plt.legend()
        plt.show()
    
    def clear(self):
        self.train_acc = []
        self.test_acc = []
        self.train_loss = []
        self.test_loss = []