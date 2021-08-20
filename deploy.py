'''
Author  : Zhengwei Li
Version : 1.0.0 
'''

import time
import cv2
import torch 
import pdb
import argparse
import os 
import numpy as np
import torch.nn.functional as F
import pdb
import copy
from model.segnet import SegMattingNet, SegMattInferNet

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')
parser.add_argument('--model', default='./model/*.pt', help='preTrained model')
parser.add_argument('--inputPath', default='./', help='input data path')
parser.add_argument('--savePath', default='./', help='output data path')
parser.add_argument('--size', type=int, default=128, help='net input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')

args = parser.parse_args()

if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

def load_model(args):
    print('Loading model from {}...'.format(args.model))

    # Using cpu only
    myModel = SegMattInferNet()
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    myModel.load_state_dict(state_dict, strict=False)
    myModel.eval()
    print(myModel)

    # Convert to mobile 
    convert_mobile(myModel, args)

    return myModel

def convert_mobile(model, args):
    example = torch.FloatTensor(1, 3, args.size, args.size)
    print(example.shape)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("jit_model.pt")

def seg_process(args, net):

    filelist = [f for f in os.listdir(args.inputPath)]
    filelist.sort()

    # set grad false
    torch.set_grad_enabled(False)
    i = 0.001
    t_all = 0

    directory = args.savePath
    if not os.path.exists(directory):
        os.makedirs(directory)

    for f in filelist:

        print('The %dth image : %s ...'%(i,f))

        image = cv2.imread(os.path.join(args.inputPath, f)) 
        # image = image[:,400:,:]
        origin_h, origin_w, c = image.shape
        image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
        image_resize = (image_resize - (104., 112., 121.,)) / 255.0   
        print(image_resize.shape)     

        tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
        tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
        inputs = tensor_4D.to(device)

        t0 = time.time()
        alpha = net(inputs)


        if args.without_gpu:
            alpha_np = alpha[0,0,:,:].data.numpy()
        else:
            alpha_np = alpha[0,0,:,:].cpu().data.numpy()

        tt = (time.time() - t0)

        # print(seg)
        # print(seg.shape)
        # np.savetxt(os.path.join(args.savePath, f[:-4] + '_seg_.csv'), seg.data.numpy())
        # cv2.imwrite(os.path.join(args.savePath, f[:-4] + '_seg_.png'), seg.data.numpy())

        # numpy.savetxt("foo.csv", a, delimiter=",")
        import pandas as pd
        df = pd.DataFrame(alpha_np)
        df = df.applymap(lambda x: 1.0 if x > 0.5 else 0.0)
        alpha_np = df.to_numpy()

        np.savetxt(os.path.join(args.savePath, f[:-4] + '_alpha_.csv'), alpha_np)

        cv2.imwrite(os.path.join(args.savePath, f[:-4] + '_alpha_.png'), alpha_np*255)



        alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
        print(alpha_np)
        print(alpha_np.shape)
        
        

        seg_fg = np.multiply(alpha_np[..., np.newaxis], image)

        f = f[:-4] + '_.png'
        cv2.imwrite(os.path.join(args.savePath, f), seg_fg)

        i+=1
        t_all += tt
        # break

    print("image number: {} mean matting time : {:.0f} ms".format(i, t_all/i*1000))

def main(args):

    myModel = load_model(args)
    seg_process(args, myModel)

if __name__ == "__main__":
    main(args)
