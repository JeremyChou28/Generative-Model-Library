'''
Description: auto-encoder model
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2022-12-07 21:42:44
'''
from torch import nn


class AE(nn.Module):

    def __init__(self):
        # 调用父类方法初始化模块的state
        super(AE, self).__init__()

        # 编码器 ： [b, 784] => [b, 20]
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, 20), nn.ReLU())

        # 解码器 ： [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # 图片数值取值为[0,1]，不宜用ReLU
        )

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, 784)  # 一行代表一个样本

        # encoder
        x = self.encoder(x)

        # decoder
        x = self.decoder(x)

        # reshape
        x = x.view(batch_size, 1, 28, 28)
        return x
