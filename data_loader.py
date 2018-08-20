import os
import os.path
import sys
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class TransformVOCDetectionAnnonation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj[4]
            bndbox = [int(bb.text)-1 for bb in bbox]

            res += [bndbox + [name]]

        return res


class PascalVOC(Dataset):

    def __init__(self,
                 data_path,
                 dataset,
                 mode,
                 transform=None,
                 target_transform=None):
        """
        Initialize dataset
        """

        self.data_path = data_path
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        self.annotation_path = os.path.join(self.data_path,
                                            dataset,
                                            'Annotations',
                                            '%s.xml')
        self.image_path = os.path.join(self.data_path,
                                       dataset,
                                       'JPegImages',
                                       '%s.jpg')
        self.text_path = os.path.join(self.data_path,
                                      dataset,
                                      'ImageSets',
                                      'Main',
                                      '%s.txt')

        with open(self.text_path % self.mode) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

        self.N = len(self.ids)

    def __len__(self):
        """
        Number of data in the dataset
        """
        return self.N

    def __getitem__(self, index):
        """
        Return item from dataset
        """
        image_id = self.ids[index]

        target = ET.parse(self.annotation_path % image_id).getroot()

        image = Image.open(self.image_path % image_id).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def show(self, index):
        """
        displays the image
        """
        image, target = self.__getitem__(index)
        draw = ImageDraw.Draw(image)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        image.show()


def get_loader(data_path, mode='train'):
    """
    Get dataset loader
    """
    pass
