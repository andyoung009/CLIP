import os
import torch
import pandas as pd
from PIL import Image
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")
import clip

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class IN2POSEDATASET(Dataset):
    # def __init__(self, root='/data/ML_document/datasets/custom_6dpose_dataset', ann_file='custom_6dpose_dataset.csv', transform=transforms.ToTensor(), target_transform=None):
    # def __init__(self, root='/data/ML_document/datasets/custom_6dpose_dataset', ann_file='custom_6dpose_dataset.csv', transform=None, target_transform=None):
    def __init__(self, root='/data/ML_document/datasets/custom_6dpose_dataset', ann_file='custom_6dpose_dataset_mask.csv', transform=transforms.ToTensor(), target_transform=None):        

        super(IN2POSEDATASET, self).__init__()

        self.data_path = root
        self.ann_path = os.path.join(self.data_path, ann_file)
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = pd.read_csv(self.ann_path)

    def __len__(self):
        return len(self.database)

    def _load_images(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def _transformer_img(self, image):
        _, pre_transformer = clip.load("RN50", device=device)
        img_trans = pre_transformer(image)
        return img_trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # idb = self.database[index]

        # instructions & transfer tuple to list
        instructions = self.database.iloc[index, 0]
        # instructions = list(instructions)
        # idb.iloc[0]

        # images(rgb&depth)
        # images_rgb = self._load_images(self.data_path + '/' + self.database.iloc[index, 1]).convert('RGB')
        images_rgb = self._load_images(self.data_path + '/' + self.database.iloc[index, 1])
        # if self.transform is not None:
        #     images_rgb = self.transform(images_rgb)
        images_rgb = self._transformer_img(images_rgb)

        images_depth = np.load(self.data_path + '/' + self.database.iloc[index, 2])
        if images_depth.dtype == np.uint16:
            # images_depth = images_depth.astype(np.uint8)
            images_depth = images_depth.astype(float)
            images_depth = np.float32(images_depth)
        # images_depth = self._load_images(self.data_path + '/' + idb[2])
        # if self.transform is not None:
        #     images_depth = self.transform(images_depth)

        mask = np.load(self.data_path + '/' + self.database.iloc[index, 6])

        # outpus  shape = (b,3+4=7)
        info_dict = eval(self.database.iloc[index, 3])
        outputs = torch.cat((torch.tensor(info_dict["translation"]),torch.tensor(info_dict["rotation"])),dim = -1)
        if self.target_transform is not None:
            outputs = self.target_transform(outputs)

        return (instructions, images_rgb, images_depth, mask, outputs)


if __name__ == '__main__':
    datasets6d2pose = IN2POSEDATASET()
    dataset = DataLoader(dataset=datasets6d2pose,batch_size=6,shuffle=True,num_workers=2)
    # dataiter = iter(dataset)
    # data = next(dataiter.__next__())
    for i ,(a,b,c,d,e) in enumerate(dataset):
    # a,b,c,d = data
        print(a,b.shape,c.shape,d.shape)
    print(datasets6d2pose.__len__())
    a, b, c, d, e  = datasets6d2pose.__getitem__(index=1)
    print('test!')
