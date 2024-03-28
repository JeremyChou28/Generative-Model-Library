'''
Description: use auto-encoder model to process mnist
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2022-12-07 21:43:04
'''
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ae import AE
from torch import nn, optim
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def main(epoch_num):
    # 下载mnist数据集
    mnist_train = datasets.MNIST(root='../dataset/mnist',
                                 train=True,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor()]),
                                 download=True)
    mnist_test = datasets.MNIST(root='../dataset/mnist',
                                train=False,
                                transform=transforms.Compose(
                                    [transforms.ToTensor()]),
                                download=True)

    # 载入mnist数据集
    # batch_size设置每一批数据的大小，shuffle设置是否打乱数据顺序，结果表明，该函数会先打乱数据再按batch_size取数据
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    # 查看每一个batch图片的规模
    x, label = iter(mnist_train).__next__()  # 取出第一批(batch)训练所用的数据集
    print(' img : ', x.shape
          )  # img :  torch.Size([32, 1, 28, 28])， 每次迭代获取32张图片，每张图大小为(1,28,28)

    # 准备工作 : 搭建计算流程
    device = torch.device('cuda')
    model = AE().to(device)  # 生成AE模型，并转移到GPU上去
    print('The structure of our model is shown below: \n')
    print(model)
    loss_function = nn.MSELoss()  # 生成损失函数
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # 开始迭代
    loss_epoch = []
    for epoch in range(epoch_num):
        # 每一代都要遍历所有的批次
        for batch_index, (x, _) in enumerate(mnist_train):
            # [b, 1, 28, 28]
            x = x.to(device)
            # 前向传播
            x_hat = model(x)  # 模型的输出，在这里会自动调用model中的forward函数
            loss = loss_function(x_hat, x)  # 计算损失值，即目标函数
            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了

        loss_epoch.append(loss.item())
        if epoch % (epoch_num // 10) == 0:
            print('Epoch [{}/{}] : '.format(epoch, epoch_num), 'loss = ',
                  loss.item())  # loss是Tensor类型
            # x, _ = iter(mnist_test).__next__()   # 在测试集中取出一部分数据
            # with torch.no_grad():
            #     x_hat = model(x)

    return loss_epoch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epoch_num = 100
    loss_epoch = main(epoch_num=epoch_num)
    # 绘制迭代结果
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
