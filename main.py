import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
#import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import time
import random
import UNet



'''
#for loading the data from kaggle to colab to run over GPUs,
#run the following

! pip install -q kaggle

from google.colab import files
files.upload()  # upload the token file downloaded from setting in your kaggle settings.

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
! kaggle competitions download -c 'Carvana-Image-Masking-Challenge'


from google.colab import drive
drive.mount('/content/gdrive')  # add the genetared key

# select the file path accordingly 
! mkdir /content/gdrive/MyDrive/Project_Data/Carvan/test_hq
! mkdir /content/gdrive/MyDrive/Project_Data/Carvan/train_hq
! mkdir /content/gdrive/MyDrive/Project_Data/Carvan/train_masks

! unzip train_hq.zip -d /content/gdrive/MyDrive/Project_Data/Carvan/train_hq
! rm train_hq.zip

! unzip train_masks.zip -d /content/gdrive/MyDrive/Project_Data/Carvan/train_masks
! rm train_masks.zip

! unzip test_hq.zip -d /content/gdrive/MyDrive/Project_Data/Carvan/test_hq
! rm test_hq.zip
'''


def CarDataloader(df, img_fol, mask_fol, batch_size):
    """load train and validation data form respective datasets"""

    df_train, df_valid = train_test_split(df, test_size=0.2)
    train_dataset = CarDataset(df_train, img_fol, mask_fol, 'train')
    valid_dataset = CarDataset(df_valid, img_fol, mask_fol, 'valid')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['train'], num_workers=16,
                                                   pin_memory=True, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size['valid'], num_workers=16,
                                                   pin_memory=True, drop_last=True)
    return train_dataloader, valid_dataloader


def dice_score(pred, targs):
    """calculate the dice score"""

    # pred = (pred>0).float()
    pred = torch.sigmoid(pred)
    return 2. * (pred * targs).sum() / ((pred ** 2) + (targs ** 2)).sum()


def transform(image, mask, phase):
    """image transformations to apply"""

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    mask = to_tensor(mask)

    Norm = transforms.Normalize(mean=mean, std=std)
    image = Norm(image)

    if phase == 'train':
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    mask = resize(mask)

    return image, mask


class CarDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_fol, mask_fol, phase):
        self.fname = df['img'].values.tolist()
        random.shuffle(self.fname)
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.phase = phase

    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        name = self.fname[idx]
        img_name_path = os.path.join(self.img_fol, name)
        mask_name_path = os.path.join(self.mask_fol, name)[:-4] + '_mask.png'

        img = cv2.imread(img_name_path)
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)

        img_aug, mask_aug = transform(img, mask, self.phase)
        return img_aug, mask_aug


class ImageSegmentation(object):
    """Main class to ttrain the model"""

    def __init__(self, model, img_fol, mask_fol, df):
        self.dataframe = df
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.phases = ['train', 'valid']
        self.batch_size = {'train': 16, 'valid': 16}
        self.train_dataloader, self.valid_dataloader = CarDataloader(df, img_fol, mask_fol, self.batch_size)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.device = torch.device("cuda:0")
        self.model = model.to(self.device)
        torch.backends.cudnn.benchmark = True

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2,
                                                                    verbose=True)
        self.best_loss = float('inf')

    def fit(self, epochs=10):
        for epoch in range(epochs):
            losses = []
            dice_loss = []
            epoch_start = time.time()
            start = time.time()
            print(f"Starting epoch: {epoch} | phase:{self.phases[0]}")
            for batch_idx, (input, targets) in enumerate(self.train_dataloader):
                input, targets = input.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(input)
                loss = self.criterion(predictions, targets)
                loss.backward()

                self.optimizer.step()
                losses.append(loss.item())
                dice_loss.append(dice_score(predictions, targets).item())
                end = time.time()

                if batch_idx % 100 == 0:
                    print('Batch Index : %d Loss : %.3f Dice Score : %.3f Time : %.3f seconds ' % (
                        batch_idx, np.mean(losses), np.mean(dice_loss), end - start))

                    start = time.time()
            print(f"Starting epoch: {epoch} | phase:{self.phases[1]}")
            torch.cuda.empty_cache()
            dice_loss = []
            losses = []
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.valid_dataloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    valid_loss = self.criterion(outputs, targets)
                    losses.append(valid_loss.item())

                    dice_loss.append(dice_score(outputs, targets).item())
                epoch_loss = np.mean(losses)
                if epoch_loss < self.best_loss:
                    print("******** New optimal found ********")
                    self.best_loss = epoch_loss

                print('Epoch : %d Test Acc : %.3f Loss : %.3f' % (epoch, np.mean(dice_loss), epoch_loss))
                print('--------------------------------------------------------------')

            self.model.train()
            self.scheduler.step(epoch_loss)

    def predict(self):
        """predictions"""

        for i, batch in enumerate(self.valid_dataloader):
            images, mask_target = batch

            batch_preds = torch.sigmoid(self.model(images.to(self.device)))
            batch_preds = batch_preds.detach().cpu().numpy()
            for pre in range(16):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
                fig.suptitle('predicted_mask  <--------------->  original_mask')
                ax1.imshow(np.squeeze(batch_preds[pre]), cmap='gray')
                ax2.imshow(np.squeeze(mask_target[pre]), cmap='gray')
                plt.show()
            break


if __name__ == '__main__':
    df = pd.read_csv('/train_masks.csv')
    img_fol = '/content/gdrive/MyDrive/Project_Data/Carvan/train/train'
    mask_fol = '/content/gdrive/MyDrive/Project_Data/Carvan/train_masks_png'

    # mod = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    mod = UNet.UNet(3, 1, False)
    Segment = ImageSegmentation(mod, img_fol, mask_fol, df)
    Segment.fit(epochs=10)
