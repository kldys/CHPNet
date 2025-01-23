import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import CHPNet
from network_swinir import SwinIR
from Dataset import Train_Data
from config import opt
from torch.utils.data import DataLoader
from PIL import Image
from skimage import io, color
from torchvision import transforms as T
import os
from skimage.metrics import structural_similarity as compare_ssim
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        tensor_gpu = torch.tensor(x).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img1 = tensor_cpu.numpy()
        tensor_gpu = torch.tensor(y).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img2 = tensor_cpu.numpy()

        edge_output1 = cv.Canny(img1.astype(np.uint8), 50, 150)
        edge_output1[edge_output1 == 255] = 1
        edge_output2 = cv.Canny(img2.astype(np.uint8), 50, 150)
        edge_output2[edge_output2 == 255] = 1
        a3 = abs(edge_output1.astype(np.float32) - edge_output2.astype(np.float32))
        edge_loss = torch.tensor(sum(a3)).cuda(0)
        #edge_loss= torch.tensor(sum(a3)).to(device)
        mse_loss = torch.mean(torch.pow((x - y), 2))+edge_loss
        return mse_loss

def train(load_model_path=None):
    f = open("log")
    train_data = Train_Data()
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model=SwinIR(upscale=opt.scaling_factor, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'
    teacher_model.eval()
    pretrained_model = torch.load(opt.teacher_model_path)
    teacher_model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    teacher_model=teacher_model.to(device)


    temp = 7
    alpha = 0.3
    soft_loss = nn.KLDivLoss(reduction='batchmean')

    net = CHPNet()
    net = net.cuda()
    net = nn.DataParallel(net)


    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    if load_model_path:
        net.load_state_dict(torch.load(load_model_path))

    criterion =CustomLoss()
    criterion = criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    num_show = 0
    psnr_best = 0

    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()
            torch.cuda.empty_cache()

            teachers_preds = teacher_model(data)
            teachers_preds = torch.mean(teachers_preds, dim=1, keepdim=True)

            optimizer.zero_grad()
            output = net(data)

            students_loss = criterion(output, label)
            ditillation_loss = soft_loss(
                F.softmax(output / temp, dim=1),
                F.softmax(teachers_preds / temp, dim=1)
            )
            loss = alpha * students_loss + (1 - alpha) * ditillation_loss

            loss.backward()
            optimizer.step()



            if i % 20 == 0:  # save parameters every 20 batches
                mse_loss, psnr_now, ssim = val(net, epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f' % (epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now, ssim))
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f' % (epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now, ssim), file=f)
                num_show += 1
                x = torch.Tensor([num_show])
                y1 = torch.Tensor([mse_loss])
                y2 = torch.Tensor([psnr_now])


                if psnr_best < psnr_now:
                    psnr_best = psnr_now
                    torch.save(net.state_dict(), opt.save_model_path)


        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
            print('learning rate: ', optimizer.param_groups[0]['lr'], file=f)
    print('Finished Training')
    print('Finished Training', file=f)
    f.close()

def val(net1, epoch, i):
    with torch.no_grad():
        psnr_ac = 0
        ssim_ac = 0
        for j in range(5):
            label = io.imread('')
            test = io.imread('')

            label_ycbcr = color.rgb2ycbcr(label)
            test_ycbcr = color.rgb2ycbcr(test)
            label_y = label_ycbcr[:, :, 0] / 255
            test_y = test_ycbcr[:, :, 0] / 255

            label_cb = label_ycbcr[:, :, 1]
            label_cr = label_ycbcr[:, :, 2]

            label = torch.FloatTensor(label_y).unsqueeze(0).unsqueeze(0).cuda()
            test = torch.FloatTensor(test_y).unsqueeze(0).unsqueeze(0).cuda()

            output = net1(test)
            output = torch.clamp(output, 0.0, 1.0)
            loss = (output*255 - label*255).pow(2).sum() / (output.shape[2]*output.shape[3])
            psnr = 10*np.log10(255*255 / loss.item())

            output = output.squeeze(0).squeeze(0).cpu()
            label = label.squeeze(0).squeeze(0).cpu()

            output_array = np.array(output * 255).astype(np.float32)
            label_array = np.array(label * 255).astype(np.float32)
            ssim = compare_ssim(output_array, label_array, data_range=255)

            psnr_ac += psnr
            ssim_ac += ssim


        if i%400 == 0:
            SR_image = np.zeros([*label_array.shape, 3])
            SR_image[:, :, 0] = output_array
            SR_image[:, :, 1] = label_cb
            SR_image[:, :, 2] = label_cr
            save_index = str(int(epoch*(opt.num_data/opt.batch_size/400) + (i+1)/400))
            SR_image = color.ycbcr2rgb(SR_image)*255
            SR_image = np.clip(SR_image, a_min=0., a_max=255.)
            SR_image = SR_image.astype(np.uint8)
            io.imsave('')

    return loss, psnr_ac/5, ssim_ac/5


if __name__ == '__main__':
    train()





