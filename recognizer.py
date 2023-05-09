import cv2
import numpy as np

import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

class Recognizer():
    def __init__(self, name_dict) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = MTCNN(device=self.device)
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.name_dict = name_dict
        self.embedding_data = []
    
    
    def create_embeddings(self, loader):
        faces = []
        names = []
        for x, y in loader:
            ret, box = self.detect_face(x)
            if ret:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
                face = x.crop((x1, y1, x2, y2))
                
                face = face.resize((160, 160))
                face = transforms.ToTensor()(face)
                
                faces.append(face)
                names.append(self.name_dict[y])
        
        faces = torch.stack(faces).to(self.device)
        result = self.facenet(faces).detach().cpu()
        
        for e, n in zip(result, names):
            data = (e,n)
            self.embedding_data.append(data)
            
            
    def detect_face(self, image):
        boxes, prob = self.mtcnn.detect(image)
        if boxes is not None:
            box = list(map(int, boxes[0]))
            return True, box
        else:
            return False, None
    
    
    def inference(self, image, threshold=1):
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0).to(self.device)

        result = self.facenet(image).detach().cpu()
        
        comparison, names = [], []
        for data in self.embedding_data:
            dist = (result - data[0]).norm().item()
            comparison.append(dist)
            names.append(data[1])
        
        if min(comparison) < threshold:
            name = names[np.argmin(comparison)]
            return name
        else:
            return "Guest"
        
if __name__ == "__main__":
        
    from dataset import faceDataset
    from torch.utils.data import DataLoader

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((512,512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    dataset = faceDataset(path="cyl_dataset", transform=None)

    dataloader = DataLoader(dataset=dataset, collate_fn=lambda x: x[0])


    recognizer = Recognizer(name_dict=dataset.get_label_dict())
    recognizer.create_embeddings(dataloader)


    # Inference
    image = cv2.imread("face_dataset/Penny/Penny.jpg")
    # image = cv2.imread("cyl_dataset/Ken/Ken.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, box = recognizer.detect_face(image=image)
    if ret:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        result = recognizer.inference(face)
        print(result)

