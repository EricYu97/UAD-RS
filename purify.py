# from diffusers import DDPMPipeline, DDIMPipeline
from pipeline_conditional_noise import DDPMPipeline
from diffusers import DDPMScheduler
from PIL import Image
import torch
import numpy as np
import tqdm
import torchvision.transforms as tfs
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from torchvision import transforms as tfs

ANLS_Trial_Samples = 500

class test_loader(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.path=path
        self.filelist=os.listdir(path)
    def __getitem__(self, index):
        im=Image.open(os.path.join(self.path,self.filelist[index]))
        return tfs.ToTensor()(im),self.filelist[index]
    def __len__(self):
        return len(self.filelist)


def args():
    parser = argparse.ArgumentParser(description='ANLS Scheme')
    parser.add_argument('--input_path', type=str, default='/home/yu34/AdvD/UAE-RS/UCM_adv/test/fgsm/alexnet', help='path for input data (adversarial examples)')
    parser.add_argument('--pretrained_path', type=str, default='/home/yu34/AdvD/diffusers/ddpm-ema-UCM-256-1k-1200steps/checkpoint-17000', help='path for pretrained model')
    parser.add_argument('--save_path', type=str, default='/home/yu34/AdvD/UAE-RS/UCM_1200steps/fgsm/alexnet/1ksteps_noise_level_10/', help='path for saving the purified results')
    parser.add_argument('--phase', type=int, default='0', help='0 for adversarial purify from all noise levels, 1 for phase after ANLS')
    parser.add_argument('--noise_level', type=int, default='0', help='optimal noise level obtained from ANLS')
    args = parser.parse_args()
    return args

def defense(input_path,pretrained_path,save_path,noise_level):
    os.makedirs(save_path,exist_ok=True)
    device="cuda"
    noise_scheduler = DDPMScheduler(num_train_timesteps=1200)
    trained_model=DDPMPipeline.from_pretrained(pretrained_path).unet.to("cuda")
    # trained_model=torch.nn.DataParallel(trained_model,device_ids=[0,1,2,3,4,5,6,7])
    noise_level =noise_level
    def show(tensor):
        return Image.fromarray((tensor.cpu().permute(0, 2, 3, 1) * 255.0).type(torch.uint8).numpy()[0])
    def noise(img, t):
        noise = torch.randn(img.shape).to(device)
        timesteps = torch.LongTensor([t]).to(device)
        print(img.device,noise.device)
        noisy_image = noise_scheduler.add_noise(img, noise, timesteps)
        return noisy_image

    def imageopen(path):
        img=Image.open(path)
        tensor=tfs.ToTensor()(img)
        tensor=tensor.unsqueeze(0)
        return tensor
    
    @torch.no_grad()
    def denoise(img, bs, ts, progress=True):
        noise_scheduler.set_timesteps(1200)
        img_reconstruct = img.reshape(bs, 3, 256, 256).to("cuda")
        

        for t in noise_scheduler.timesteps[-ts:]:
            # 1. predict noise model_output
            # print(img_reconstruct.shape, t)
            
            
            model_output = trained_model(img_reconstruct, t).sample


            # 2. compute previous img_reconstruct: x_t -> x_t-1
            img_reconstruct = noise_scheduler.step(model_output, t, img_reconstruct).prev_sample

        return img_reconstruct
    
    batch_size=50
    testloader=DataLoader(dataset=test_loader(path=input_path),batch_size=50,shuffle=True,num_workers=1,pin_memory=False)

    for i,batch in enumerate(testloader):
        sample_images, sample_image_names= batch
        sample_images=sample_images.to(device)
        noisy_image=noise(sample_images,noise_level)
        (bs, _, _, _)=sample_images.shape
        imgs_reconstruct=denoise(noisy_image, bs, noise_level)
        for n in range(bs):
            img_reconstruct=imgs_reconstruct[n,:,:,:].unsqueeze(0)
            reconstructed_img=show(img_reconstruct)
            reconstructed_img.save(f'{save_path}/{sample_image_names[n].split("_adv")[0]}_reconstructed.png')

        if (i+1)*batch_size > ANLS_Trial_Samples:
            break # Set 500 as the trial samples, to facilitate the ANLS computation.


if __name__=="__main__":
    args=args()
    if args.phase==0:
    #ANLS Phase
        for noise_level in range(0,151,10):
            defense(args.input_path,args.pretrained_path,args.save_path,noise_level=noise_level)

    #Purify all
    if args.phase==1:
        noise_level=50 #Adjust it to your ANLS results.
        defense(args.input_path,args.pretrained_path,args.save_path,noise_level=args.noise_level)    