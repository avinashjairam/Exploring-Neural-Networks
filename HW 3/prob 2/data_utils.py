from tqdm import tqdm
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

        self.class_to_idx = {
            'bald_eagle': 0, 'black_bear': 1,
            'bobcat': 2, 'canada_lynx': 3,
            'columbian_black-tailed_deer': 4, 'cougar': 5,
            'coyote': 6, 'deer': 7,
            'elk': 8, 'gray_fox': 9,
            'gray_wolf': 10, 'mountain_beaver': 11,
            'nutria': 12, 'raccoon': 13,
            'raven': 14, 'red_fox': 15,
            'ringtail': 16, 'sea_lions': 17,
            'seals': 18, 'virginia_opossum': 19
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
    filepaths = ['data/black_bear/1.jpg',
                 'data/black_bear/2.jpg',
                 'data/bald_eagle/1.jpg',
                 'data/bald_eagle/6b9b6fa9e5c3c4e803.png']
    dataset = MyDataset(filepaths=filepaths)
