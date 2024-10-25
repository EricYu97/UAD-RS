from diffusers import DDPMScheduler, UNet2DModel
# from pipeline_conditional_noise import DDPMPipeline
from PIL import Image
import torch
import numpy as np
from diffusers import DDPMPipeline

scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline=DDPMPipeline.from_pretrained("/home/yu34/AdvD/diffusers/ddpm-ema-AID-256-1k/checkpoint-79000/").to("cuda")
scheduler.set_timesteps(1000)

sample_size = 256

i=0
for ii in range(10):
    noise = torch.randn((50, 3, sample_size, sample_size)).to("cuda")
    input = noise

    with torch.no_grad():
            generator = torch.Generator(device=pipeline.device).manual_seed(ii)
                # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                    generator=generator,
                    batch_size=50,
                    num_inference_steps=1000,
                    output_type="pil",
                ).images

    for image in images:
        image.save("./image_generation_results/AID/standard/"+str(i)+".png")
        i+=1

    # for t in scheduler.timesteps:
        
            # noisy_residual = model(input, t).sample
            # prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            # input = prev_noisy_sample
    # print(images.shape())
    # for j in range(images.shape[0]):
    #     # image = (images[j:j+1,:,:,:] / 2 + 0.5).clamp(0, 1)
    #     # image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    #     image = Image.fromarray((image * 255).round().astype("uint8"))
    #     image.save("./image_generation_results/UCM/standard/"+str(i)+".png")
    #     i+=1