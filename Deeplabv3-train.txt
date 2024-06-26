import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from ctypes import sizeof
import numpy as np
import pandas as pd
import time
import cv2
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import albumentations as A
from rasterio.windows import Window
import segmentation_models_pytorch as smp


import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
from torchvision import transforms as T



def rle_encode(im):  # mask->img
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):  # img->mask
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')



EPOCHES = 10  #训练轮次
BATCH_SIZE = 2     #批量大小
IMAGE_SIZE = 256        #图片像素规格
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据增广
trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        # 添加更多的增强方法
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.3),
        A.Blur(blur_limit=(3, 7), p=0.3),
    ], p=0.3),
])


# 图片数据编码处理成tensor格式
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])  # 获取图片mask矩阵
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_model():
    model = smp.DeepLabV3(
        encoder_name="resnet50",  # 选择编码器，例如resnet50
        encoder_weights='imagenet',  # 使用ImageNet预训练权重进行编码器初始化
        in_channels=3,  # 模型输入的通道数，RGB图像为3
        classes=1,  # 模型输出的通道数，二分类问题为1
    )
    return model


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        # output = model(image)['out']
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits  # 如果BCE带logits，则损失函数在计算BCEloss之前，自动计算softmax/sigmoid将其映射到[0,1]
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=True)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if (self.reduce):
            return torch.mean(F_loss)
        else:
            return F_loss


# bce:dice  0.8:0.2
def loss_fn(y_pred, y_true):
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = FocalLoss()
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice


def run_model1(model):
    best_loss = 10
    count = 1
    for epoch in range(1, EPOCHES + 1):
        count = 0
        losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm_notebook(loader):
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()
            # output = model(image)['out']
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(count)
            count = count + 1

        vloss = validation(model, vloader, loss_fn)
        scheduler.step(vloss)

        print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))
        losses = []

        # 保存模型
        if vloss < best_loss:  # 若验证集的验证loss比当前最小loss还小
            best_loss = vloss
            torch.save(model.state_dict(), '{}_unet_model_best_{}.pth'.format(epoch, 1 - vloss))


if __name__ == '__main__':
    train_mask = pd.read_csv(r'D:\data_of_tianchi\train_mask.csv\train_mask.csv', sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: 'D:/data_of_tianchi/train/train/' + x)  # 图片读取到数组train_mask中

    dataset = TianChiDataset(  # trfm是什么？
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        trfm, False
    )

    # 划分训练集和验证集6：1
    valid_idx, train_idx = [], []
    for i in range(len(dataset)):
        if i % 7 == 4:
            valid_idx.append(i)
        # elif i % 7==1:
        else:
            train_idx.append(i)

    train_ds = D.Subset(dataset, train_idx)
    valid_ds = D.Subset(dataset, valid_idx)
    print("train len:", len(train_ds))
    print("valid len:", len(valid_ds))

    # 定义训练和验证的loader
    loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    vloader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 创建模型
    model = get_model()
    model.to(DEVICE)
    # model.load_state_dict(torch.load("fold2_unet_model_new_0.898655166849494.pth"),False)
    model.eval()

    # optimizer优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1)

    # 训练模型
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m 
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)

    run_model1(model)
    # run_model2(model2)