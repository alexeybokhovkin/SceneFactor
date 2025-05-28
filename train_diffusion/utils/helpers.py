import math
from inspect import isfunction
import os

import torch


def save_model(iters, model, optimizer, loss, path):
    torch.save({'iters': iters,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None
    
    model.load_state_dict(checkpoint['model_state_dict'])

    # model_ckpt = checkpoint['model_state_dict']
    # model_dict = OrderedDict()
    # for k,v in model_ckpt.items():
    #     model_dict['module.' + k] = v
    # model.load_state_dict(model_dict)

    loss = checkpoint['loss']
    iters = checkpoint['iters']
    print("loading from iter {}...".format(iters))
    return iters, model, optimizer, loss


def save_code_to_conf(conf_dir):
    path = os.path.join(conf_dir, "code")
    os.makedirs(path, exist_ok=True)
    for folder in ["utils"]: 
        os.makedirs(os.path.join(path, folder), exist_ok=True)
        os.system("""cp -r ./{0}/* "{1}" """.format(folder, os.path.join(path, folder)))

    # other files
    os.system("""cp *.py "{}" """.format(path))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

#from 0,1 to -1,1
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# from -1,1 to 0,1
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# from any batch to [0,1]
# f should have shape (batch, -1)
def normalize_to_zero_to_one(f):
    f -= f.min(1, keepdim=True)[0]
    f /= f.max(1, keepdim=True)[0]
    return f


# extract the appropriate t index for a batch of indices
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    #print("using LINEAR schedule")
    scale = 1000 / timesteps
    # beta_start = scale * 0.0001 
    # beta_end = scale * 0.02 
    betas = (
            torch.linspace(0.00003 ** 0.5, 0.01 ** 0.5, timesteps, dtype=torch.float64) ** 2
    )
    return betas

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    #print("using COSINE schedule")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


