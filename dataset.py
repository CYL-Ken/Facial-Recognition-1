import os
import piexif
from PIL import Image
from torch.utils.data.dataset import Dataset

class faceDataset(Dataset):
    def __init__(self, path, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform
        
        self.label = []
        self.image_path = []
        
        self.name = []
        for person in os.listdir(path):
            if person not in self.name:
                self.name.append(person)
                
            person_dataset = os.path.join(path, person)
            for i in os.listdir(person_dataset):
                self.image_path.append(os.path.join(person_dataset, i))
                self.label.append(person)
            
        self.one_hot = {k: v for v, k in enumerate(self.name)}
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
        path = self.image_path[index]
        piexif.remove(path)
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        label = self.one_hot[self.label[index]]
        
        return image, label
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.image_path)
    
    def get_label_dict(self):
        return {k: v for k, v in enumerate(self.name)}