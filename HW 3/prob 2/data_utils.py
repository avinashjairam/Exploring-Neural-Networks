from tqdm import tqdm
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

        self.class_to_idx = {'frog': 0, 'truck': 1,
                             'deer': 2, 'automobile': 3,
                             'bird': 4, 'horse': 5,
                             'ship': 6, 'cat': 7,
                             'dog': 8, 'airplane': 9}

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

            # resize
            image = image.resize((128, 128), PIL.Image.ANTIALIAS)

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
    filepaths = ['data/frog/1.jpg', 'data/frog/2.jpg', 'data/deer/1.jpg']
    dataset = MyDataset(filepaths=filepaths)
