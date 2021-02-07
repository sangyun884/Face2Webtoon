from torchvision import transforms
import numpy as np
import PIL
import random
import torch
import utils

class ADA:
    def __init__(self, xflip=1, rot90=1, int_translation=1, iso_scale=1, abrot=1, aniso_scale=1, frac_translation = 1, bright = 1, contrast = 1, lumaflip = 1, huerot = 1, sat = 1, target_value = 0.6):
        self.xflip = xflip
        self.rot90 = rot90
        self.int_translation = int_translation
        self.iso_scale = iso_scale
        self.abrot = abrot
        self.aniso_scale = aniso_scale
        self.frac_translation = frac_translation
        self.bright = bright
        self.contrast = contrast
        self.lumaflip = lumaflip
        self.huerot = huerot
        self.sat = sat

        self.target_value = target_value
        self.strength = 0
        self.tune_kimg = 500
        self.nimg_delta = 256
        self.output_list = []
        self.rt = 0

    def augment_pipeline(self, img):

        #Pixel blitting
        img = x_flip(img, self.xflip*self.strength)

        img = random_rotation90(img, self.rot90*self.strength)
        img = integer_translation(img, self.int_translation*self.strength)

        #Geometric transformation
        img = isotropic_scaling(img, self.iso_scale*self.strength)
        prot = 1-(1-self.strength)**0.5
        img = arbitrary_rotation(img, self.abrot*prot)#Pre-rotation
        img = anisotropic_scaling(img, self.aniso_scale*self.strength)
        img = arbitrary_rotation(img, self.abrot * prot)#Post-rotation
        img = fractional_translation(img, self.frac_translation*self.strength)

        img = padwhite(img)

        #Color transformation
        img = brightness(img, self.bright*self.strength)
        img = contrast(img, self.contrast*self.strength)
        img = luma_flip(img, self.lumaflip*self.strength)
        img = hue_rotation(img, self.huerot*self.strength)
        img = saturation(img, self.sat*self.strength)

        return img
    def calculate_rt(self,d_real_output):
        if not isinstance(d_real_output, np.ndarray):
            d_real_output = np.array(d_real_output)
        ncorrect = np.sum(d_real_output>0.5)
        nwrong = d_real_output.shape[0] - ncorrect
        rt = np.maximum((ncorrect - nwrong) / d_real_output.shape[0],0)
        return rt


    def tune(self):#Updates augmentation strength
        #d_real_output is list of discriminator output tensor. Each tensor's shape is [1,]
        rt = self.calculate_rt(self.output_list)
        self.rt = rt
        nimg_ratio = self.nimg_delta / (self.tune_kimg*1000)
        strength = self.strength
        strength += nimg_ratio*np.sign(rt - self.target_value)
        strength = np.maximum(0,strength)
        self.strength = strength
        self.output_list = []

    def feed(self, d_real_output: torch.tensor):
        self.output_list.append(d_real_output)
        if len(self.output_list)==self.nimg_delta:
            self.tune()

def x_flip(img,p):
    return transforms.RandomHorizontalFlip(p=p)(img)

def random_rotation90(img: PIL.Image, p):
    if random.random() < 1 - p:
        return img
    angle_list = [0,-90,-180,90]
    angle = random.choice(angle_list)

    return transforms.functional.rotate(img, angle)




def integer_translation(img, p, r=0.125):
    if random.random()<1-p:
        return img
    tx = np.random.uniform(-r, r)
    ty = np.random.uniform(-r, r)
    if isinstance(img, PIL.Image.Image):
        H = img.size[0]
        W = img.size[1]
    elif torch.is_tensor(img):
        H = img.size()[-2]
        W = img.size()[-1]
    img = transforms.functional.affine(img, angle=0, translate = [int(H*tx), int(W*ty)], scale=1, shear=0)
    return img
def isotropic_scaling(img, p):
    p *=2
    if random.random() < 1 - p:
        return img
    s = np.random.lognormal(0, 0.2*np.log(2))
    if s<1:
        s=2-s
    H,W = (img.size()[-2], img.size()[-1])
    img = transforms.functional.resize(img, (int(H*s), int(W*s)))
    img = transforms.functional.center_crop(img, (H,W))
    return img
def arbitrary_rotation(img, p):
    if random.random() < 1 - p:
        return img
    img = transforms.RandomRotation((-180,180))(img)
    return img
def anisotropic_scaling(img,p):
    if random.random() < 1 - p:
        return img
    s = np.random.lognormal(0, 0.2 * np.log(2))
    if s<1:
        s=2-s

    H, W = (img.size()[-2], img.size()[-1])
    if random.random()>0.5:
        img = transforms.functional.resize(img, (int(H * s), W))
    else:
        img = transforms.functional.resize(img, (H, int(W*s)))
    img = transforms.functional.center_crop(img, (H, W))
    return img
def fractional_translation(img,p, r=0.125):
    if random.random()<1-p:
        return img
    tx = np.random.uniform(-r, r)
    ty = np.random.uniform(-r, r)
    if isinstance(img, PIL.Image.Image):
        H = img.size[0]
        W = img.size[1]
    elif torch.is_tensor(img):
        H = img.size()[-2]
        W = img.size()[-1]
    img = transforms.functional.affine(img, angle=0, translate = [H*tx, W*ty], scale=1, shear=0)
    return img
def brightness(img,p):
    if random.random() < 1 - p:
        return img
    return img
def contrast(img,p):
    if random.random() < 1 - p:
        return img
    return img
def luma_flip(img,p):
    if random.random() < 1 - p:
        return img
    return img
def hue_rotation(img,p):
    if random.random() < 1 - p:
        return img
    return img
def saturation(img,p):
    if random.random() < 1 - p:
        return img
    return img
def padwhite(img):
    mask = img==0
    img_clone = img.clone()
    img_clone[mask]+=1
    return img_clone
def tensor2pil(t):
    img = transforms.functional.to_pil_image(utils.denorm(t))
    return img

def test():
    img = PIL.Image.open(r'C:\ML\face2webtoon\UGATIT-pytorch\dataset\video2anime\trainB\0\12.jpg')

    #img = transforms.functional.pil_to_tensor(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))(img)
    ada = ADA()
    ada.strength=1
    img = ada.augment_pipeline(img)
    img = tensor2pil(img)
    img.show()
