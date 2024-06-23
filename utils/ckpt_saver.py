import os
import torch

class CkptSaver():
    def __init__(self, net, tag="models", base_path="ckpt"):
        self.net = net
        self.path = os.path.join(base_path, str(tag))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, epoch):
        filename = os.path.join(self.path, "model_epoch{}.pth".format(epoch))
        torch.save(self.net.state_dict(), filename)



