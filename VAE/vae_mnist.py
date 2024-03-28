import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from vae import VAE
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import numpy as np

# plt.style.use("ggplot")

# 设置模型运行的设备
torch.cuda.set_device(1)  # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置默认参数
parser = argparse.ArgumentParser(
    description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--result_dir',
                    type=str,
                    default='./VAEResult',
                    metavar='DIR',
                    help='output directory')
parser.add_argument('--save_dir',
                    type=str,
                    default='./checkPoint',
                    metavar='N',
                    help='model saving directory')
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='batch size for training(default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    metavar='N',
                    help='number of epochs to train(default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed(default: 1)')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    metavar='PATH',
                    help='path to latest checkpoint(default: None)')
parser.add_argument('--test_every',
                    type=int,
                    default=10,
                    metavar='N',
                    help='test after every epochs')
parser.add_argument('--num_worker',
                    type=int,
                    default=1,
                    metavar='N',
                    help='the number of workers')
parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='learning rate(default: 0.001)')
parser.add_argument('--z_dim',
                    type=int,
                    default=20,
                    metavar='N',
                    help='the dim of latent variable z(default: 20)')
parser.add_argument('--input_dim',
                    type=int,
                    default=28 * 28,
                    metavar='N',
                    help='input dim(default: 28*28 for MNIST)')
parser.add_argument('--input_channel',
                    type=int,
                    default=1,
                    metavar='N',
                    help='input channel(default: 1 for MNIST)')
args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def dataloader(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 下载mnist数据集
    mnist_train = datasets.MNIST('mnist',
                                 train=True,
                                 transform=transform,
                                 download=True)
    mnist_test = datasets.MNIST('mnist',
                                train=False,
                                transform=transform,
                                download=True)

    # 载入mnist数据集
    # batch_size设置每一批数据的大小，shuffle设置是否打乱数据顺序，结果表明，该函数会先打乱数据再按batch_size取数据
    # num_workers设置载入输入所用的子进程的个数
    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return mnist_test, mnist_train, classes


def loss_function(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    loss = BCE + KLD
    return loss, BCE, KLD


def save_checkpoint(state, is_best, outdir):
    """
    每训练一定的epochs后， 判断损失函数是否是目前最优的，并保存模型的参数
    :param state: 需要保存的参数，数据类型为dict
    :param is_best: 说明是否为目前最优的
    :param outdir: 保存文件夹
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(
        outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def test(model, optimizer, mnist_test, epoch, best_test_loss):
    test_avg_loss = 0.0
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(test_x)
            # 损害函数值
            test_loss, test_BCE, test_KLD = loss_function(
                test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(mnist_test.dataset)
        '''测试随机生成的隐变量'''
        # 随机从隐变量的分布中取隐变量
        z = torch.randn(args.batch_size,
                        args.z_dim).to(device)  # 每一行是一个隐变量，总共有batch_size行
        # 对隐变量重构
        random_res = model.decode(z).view(-1, 1, 28, 28)
        # 保存重构结果
        save_image(random_res,
                   './%s/random_sampled-%d.png' % (args.result_dir, epoch + 1))
        '''保存目前训练好的模型'''
        # 保存模型
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint(
            {
                'epoch': epoch,  # 迭代次数
                'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
                'state_dict': model.state_dict(),  # 当前训练过的模型的参数
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            args.save_dir)

        return best_test_loss


def main():
    # Step 1: 载入数据
    mnist_test, mnist_train, classes = dataloader(args.batch_size,
                                                  args.num_worker)

    # 查看每一个batch图片的规模
    x, label = iter(mnist_train).__next__()  # 取出第一批(batch)训练所用的数据集
    print(
        ' img : ', x.shape
    )  # img :  torch.Size([batch_size, 1, 28, 28])， 每次迭代获取batch_size张图片，每张图大小为(1,28,28)

    # Step 2: 准备工作 : 搭建计算流程
    model = VAE(z_dim=args.z_dim).to(device)  # 生成AE模型，并转移到GPU上去
    print('The structure of our model is shown below: \n')
    print(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # Step 3: optionally resume(恢复) from a checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Step 4: 开始迭代
    loss_epoch = []
    for epoch in range(start_epoch, args.epochs):

        # 训练模型
        # 每一代都要遍历所有的批次
        loss_batch = []
        for batch_index, (x, _) in enumerate(mnist_train):
            # x : [b, 1, 28, 28], remember to deploy the input on GPU
            x = x.to(device)

            # 前向传播
            x_hat, mu, log_var = model(x)  # 模型的输出，在这里会自动调用model中的forward函数
            loss, BCE, KLD = loss_function(x_hat, x, mu,
                                           log_var)  # 计算损失值，即目标函数
            loss_batch.append(loss.item())  # loss是Tensor类型

            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了

            # print statistics every 100 batch
            if (batch_index + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                    .format(epoch + 1, args.epochs, batch_index + 1,
                            len(mnist_train.dataset) // args.batch_size,
                            loss.item() / args.batch_size,
                            BCE.item() / args.batch_size,
                            KLD.item() / args.batch_size))

            if batch_index == 0:
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat(
                    [x.view(-1, 1, 28, 28),
                     x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(
                    x_concat,
                    './%s/reconstructed-%d.png' % (args.result_dir, epoch + 1))

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(
            np.sum(loss_batch) /
            len(mnist_train.dataset))  # len(mnist_train.dataset)为样本个数

        # 测试模型
        if (epoch + 1) % args.test_every == 0:
            best_test_loss = test(model, optimizer, mnist_test, epoch,
                                  best_test_loss)
    return loss_epoch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loss_epoch = main()
    # 绘制迭代结果
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
