import os
from PIL import Image
import random



attacks=["fgsm","ifgsm","tpgd","mixcut","mixup","cw","jitter"]
models=["linknet","unet","pspnet","fcn8s"]

list_zurich=["zh16.png","zh17.png","zh18.png","zh19.png","zh20.png"]
list_vaihingen=["top_mosaic_09cm_area11.png","top_mosaic_09cm_area15.png","top_mosaic_09cm_area28.png","top_mosaic_09cm_area30.png","top_mosaic_09cm_area34.png"]

def cropimages(data_path,save_path,dataset_list):
    filelist=os.listdir(data_path)
    print(filelist)
    # generate_samples=overlap_rate
    overlap_rate=150
    total_count=0

    for file in filelist:
        if file.replace("_adv","") not in dataset_list:
            break
        count=0
        full_path=data_path+file
        img=Image.open(full_path)
        (w,h)=img.size
        print(w,h)
        i= (w-256)//overlap_rate
        j= (h-256)//overlap_rate
        for x in range(i):
            for y in range(j):
                crop_img=img.crop((overlap_rate*x,overlap_rate*y,256+overlap_rate*x,256+overlap_rate*y))
                crop_img.save(save_path+file.split(".")[0].replace("_adv","")+"_"+str(count)+".png")
                count+=1
        for x in range(i):
            crop_img=img.crop((overlap_rate*x,h-256,overlap_rate*x+256,h))
            crop_img.save(save_path+file.split(".")[0].replace("_adv","")+"_"+str(count)+".png")
            count+=1
        for y in range(j):
            crop_img=img.crop((w-256,y*overlap_rate,w,y*overlap_rate+256))
            crop_img.save(save_path+file.split(".")[0].replace("_adv","")+"_"+str(count)+".png")
            count+=1
        crop_img=img.crop((w-256,h-256,w,h))
        crop_img.save(save_path+file.split(".")[0].replace("_adv","")+"_"+str(count)+".png")
        count+=1
        print(f'generated {count} samples')
        total_count+=count
    print(f"generated total {total_count} samples.")

        # for i in range(generate_samples):
        #     rand_w=random.randint(0,w-256)
        #     rand_h=random.randint(0,h-256)
        #     crop_img=img.crop((rand_w,rand_h,rand_w+256,rand_h+256))
        #     crop_img.save(save_path+file.split(".")[0]+"_"+str(i)+".png")

if __name__=="__main__":
    # for attack in attacks:
    #     for model in models:
    #         cropimages(data_path=f"/home/yu34/AdvD/UAE-RS/segmentation/Vaihingen_adv/{attack}/{model}/",save_path=f"./segmentation/Vaihingen/cropped_test/adv/{attack}/{model}/",dataset_list=list_vaihingen)
    cropimages(data_path="/home/yu34/datasets/segmentation/Zurich/gt/",save_path="/home/yu34/datasets/segmentation/Zurich/cropped_test/gt/",dataset_list=list_zurich)
    cropimages(data_path="/home/yu34/datasets/segmentation/Zurich/img/",save_path="/home/yu34/datasets/segmentation/Zurich/cropped_test/img/",dataset_list=list_zurich)

    # cropimages(data_path="/home/yu34/datasets/segmentation/train/Zurich/gt/",save_path="/home/yu34/datasets/segmentation/train/Zurich/cropped_train/gt/",dataset_list=[])
    # cropimages(data_path="/home/yu34/datasets/segmentation/train/Zurich/img/",save_path="/home/yu34/datasets/segmentation/train/Zurich/cropped_train/img/",dataset_list=[])