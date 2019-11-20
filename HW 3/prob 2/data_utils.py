from tqdm import tqdm
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# traffic sign dataset
class TrafficSignsDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

        groupA = [0, 1, 2, 3, 4, 5, 7, 8]
        groupB = [6, 9, 10, 16, 23, 29, 32, 41, 42]
        groupC = [14, 15, 17, 18, 22, 25, 30, 31]
        groupD = [11, 12, 13, 19, 20, 21, 24, 26, 27, 28]
        groupE = [33, 34, 35, 36, 37, 38, 39, 40]

        self.class_to_idx = {}
        for orig_class_id in range(0, 42 + 1):
            if orig_class_id in groupA:
                self.class_to_idx[f'{orig_class_id}'] = 0
            elif orig_class_id in groupB:
                self.class_to_idx[f'{orig_class_id}'] = 1
            elif orig_class_id in groupC:
                self.class_to_idx[f'{orig_class_id}'] = 2
            elif orig_class_id in groupD:
                self.class_to_idx[f'{orig_class_id}'] = 3
            elif orig_class_id in groupE:
                self.class_to_idx[f'{orig_class_id}'] = 4
            else:
                raise NotImplemented

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        self.images = []
        for filepath in tqdm(self.filepaths, unit=' images', desc='prepro'):
            # open image
            image = Image.open(filepath)  # PIL image

            # resize to 128 x 128
            """
            NOTE: resizing can also be done using 
            transforms.Resize(size=(128,128)). that
            probably will not require explicit 
            conversion of RGBA images to RGB images.
            """
            image = image.resize((128, 128), PIL.Image.ANTIALIAS)

            # convert to RGB
            # (this converts RGBA images to RGB)
            # (this does not have affect on images which are already RGB)
            image = image.convert("RGB")

            # apply transforms
            image = self.transform(image)

            # append to self.images
            self.images.append(image)

        self.labels = list(
            map(
                lambda s: self.class_to_idx[s.split('/')[-2]],
                self.filepaths
            ))

    def __len__(self):
        # return size of dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# traffic sign dataset with augmentation
class TrafficSignsAugmentedDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

        groupA = [0, 1, 2, 3, 4, 5, 7, 8]
        groupB = [6, 9, 10, 16, 23, 29, 32, 41, 42]
        groupC = [14, 15, 17, 18, 22, 25, 30, 31]
        groupD = [11, 12, 13, 19, 20, 21, 24, 26, 27, 28]
        groupE = [33, 34, 35, 36, 37, 38, 39, 40]

        self.class_to_idx = {}
        for orig_class_id in range(0, 42 + 1):
            if orig_class_id in groupA:
                self.class_to_idx[f'{orig_class_id}'] = 0
            elif orig_class_id in groupB:
                self.class_to_idx[f'{orig_class_id}'] = 1
            elif orig_class_id in groupC:
                self.class_to_idx[f'{orig_class_id}'] = 2
            elif orig_class_id in groupD:
                self.class_to_idx[f'{orig_class_id}'] = 3
            elif orig_class_id in groupE:
                self.class_to_idx[f'{orig_class_id}'] = 4
            else:
                raise NotImplemented

        # tranform for orig images
        self.transform_orig = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        # transform for augmented images
        self.transform_aug = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.ColorJitter(
                            brightness=1,
                            contrast=1,
                            saturation=1,
                            hue=0.5
                        ),
                        transforms.RandomGrayscale(p=1.0),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.RandomRotation(30, resample=PIL.Image.BILINEAR),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
                    ]
                ),
                transforms.ToTensor()
            ]
        )

        self.transform = transforms.RandomChoice(
            [self.transform_orig, self.transform_aug]
        )

        self.images = []

        for filepath in tqdm(self.filepaths, unit=' images', desc='prepro'):
            # open image
            image = Image.open(filepath)  # PIL image

            # resize to 128 x 128
            """
            NOTE: resizing can also be done using 
            transforms.Resize(size=(128,128)). that
            probably will not require explicit 
            conversion of RGBA images to RGB images.
            """
            image = image.resize((128, 128), PIL.Image.ANTIALIAS)

            # convert to RGB
            # (this converts RGBA images to RGB)
            # (this does not have affect on images which are already RGB)
            image = image.convert("RGB")

            # apply transforms
            image = self.transform(image)

            # append to self.images
            self.images.append(image)

        # labels
        self.labels = list(
            map(
                lambda s: self.class_to_idx[s.split('/')[-2]],
                self.filepaths
            ))

    def __len__(self):
        # return size of dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == "__main__":
    filepaths = [
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00007.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00008.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00009.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00010.png",
        "data/gtsrb-german-traffic-sign/Train/17/00017_00000_00002.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00011.png"
    ]
    dataset = TrafficSignsDataset(filepaths=filepaths)
    dataset_withaug = TrafficSignsAugmentedDataset(filepaths=filepaths)
