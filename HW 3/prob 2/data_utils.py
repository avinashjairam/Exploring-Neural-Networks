from tqdm import tqdm
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

        self.class_to_idx = {
            f'{no}': no for no in range(42 + 1)
        }

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


if __name__ == "__main__":
    filepaths = [
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00007.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00008.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00009.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00010.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00011.png"
    ]
    dataset = MyDataset(filepaths=filepaths)
