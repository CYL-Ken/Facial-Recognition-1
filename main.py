import cv2
import json
import argparse

from torch.utils.data import DataLoader

from logger import Logger
from dataset import faceDataset
from recognizer import Recognizer
from door_controller import DoorController


def recognize_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, box = recognizer.detect_face(image=image)
    if ret:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        result = recognizer.inference(face)
        return result, (x1, y1, x2, y2)
    else:
        return None, (0,0,0,0)

def start_streaming(video_path, show_result=True):
    while True:
        video_path = config["video"] if video_path == "cyl" else video_path
        video = video_path
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            log.warning("Waiting camera...")
        
        while True:
            ret, image = cap.read()
            if not ret:
                log.warning("Cannot receive frame!")
                break
            try:
                result, (x1, y1, x2, y2) = recognize_frame(image=image)
            except Exception as e:
                # log.warning(f"Got Exception: {e}")
                result = None
                
            if args.door:
                ret, name = door.visit(result)
                if ret == True:
                    log.info(f"Hello, {name}.")
                elif name != "No Person":
                    log.info(f"Found guest!")
            
            if show_result == False:
                continue
            
            if result != None:
                cv2.rectangle(image, (x1, y1), (x2, y2), (214, 217, 8), 2, cv2.LINE_AA)
                cv2.putText(image, result, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (214, 217, 8), 2, cv2.LINE_AA)

            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            cv2.imshow("Result", image)
            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
        
        
if __name__ == "__main__":
    log = Logger().get_log()
    
    parser = argparse.ArgumentParser(description="Facial Recognition")
    parser.add_argument("-d", "--dataset", default="face_dataset", type=str, help="Dataset Path")
    parser.add_argument("-v", "--video", default=0, help="Video Stream")
    parser.add_argument("-i", "--image", default=None, help="Video Stream")
    parser.add_argument("-s", "--show", default=True, help="Show Result")
    parser.add_argument("-c", "--config")
    parser.add_argument("--door", default=False, type=bool, help="Control Door")
    args = parser.parse_args()
    
    
    log.info(f"Prepare Dataset: {args.dataset}")
    dataset = faceDataset(path=args.dataset)
    dataloader = DataLoader(dataset=dataset, collate_fn=lambda x: x[0])

    log.info(f"Create Embedding Data")
    log.info(f"Name Dict: {dataset.get_label_dict()}")
    recognizer = Recognizer(name_dict=dataset.get_label_dict())
    recognizer.create_embeddings(dataloader)
    log.info(f"Using Device: {recognizer.device}")
    
    config = json.load(open("config.json"))
    
    if args.door:
        door = DoorController(open_link = config["open"])
    
    if args.image != None:
        log.info("Image mode")
        # Not yet!
        pass
    else:
        log.info("Video mode")
        start_streaming(args.video)