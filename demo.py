# install head&body detection nn from: https://github.com/PeterH0323/Smart_Construction first!
# then put this python file just in the root dir of above repo and run!

import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from tqdm import tqdm

# hyper params
conf_thres=0.4 
iou_thres=0.5 # IOU threshold for NMS default=0.5 
classes=[0,1,2] # --class 0, or --class 0 2 3
agnostic_nms=True # class-agnostic NMS 
view_img=True
save_txt=False
names=['person', 'head', 'helmet'] # names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# Initialize detection model
device = torch_utils.select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
imgsz=2048

# Load model
model = attempt_load('helmet_head_person_s.pt', map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
print('check imsize:',imgsz)
if half:
    model.half()  # to FP16
cudnn.benchmark = True  # set True to speed up constant image size inference


def detect_and_save(
        imgfolder=r"Path/To/Folder",
        show=False
        ):
    root=os.path.split(imgfolder)[0]
    npyfolder=os.path.join(root,'Detection')
    os.makedirs(npyfolder)
    historyresult={'person':{'conf':None,'xyxy':None}, 'helmet':{'conf':None,'xyxy':None}}
    for file in tqdm(os.listdir(imgfolder)):
        basename,ext=file.split('.')
        imgpath=os.path.join(imgfolder,file)
        im0 = cv2.imread(imgpath)
        img = cv2.resize(im0, (416,416), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # [H,W,C]->[1,H,W,C]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS non_max_suppression
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # Process detections
        currentresult={'person':{'conf':None,'xyxy':None}, 'helmet':{'conf':None,'xyxy':None}}
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # det: torch.tensor(x0,y0,x1,y1, conf, cls)
                # det=tensor([[1.00000e+01, 2.07000e+02, 2.04600e+03, 2.04800e+03, 9.32226e-01, 0.00000e+00],
                #           [5.67000e+02, 2.24000e+02, 1.44200e+03, 1.21600e+03, 6.46684e-01, 2.00000e+00]], 
                #           device='cuda:0')

                # update current result dict
                for *xyxy, conf, cls in det.cpu().numpy():
                    key=names[int(cls)]
                    if key in currentresult.keys():
                        # always keep the max conf result
                        if currentresult[key]['conf'] is None or currentresult[key]['conf']<conf:
                            currentresult[key]['conf']=conf
                            currentresult[key]['xyxy']=xyxy
                    if show:
                        plot_one_box(xyxy, im0, label='%s %.2f' % (key, conf), color=colors[int(cls)], line_thickness=3)
                if show:
                    cv2.imshow("",im0)
                    cv2.waitKey(1)
            else:
                print(f"{file} detection failed!!!")
        # update history
        for key in currentresult.keys():
            # if we didn't recognize anything, we set conf=-1 and use history record.
            if currentresult[key]['xyxy'] is None:
                historyresult[key]['conf']=-1 
            else:
                historyresult[key]['conf']=currentresult[key]['conf']
                historyresult[key]['xyxy']=currentresult[key]['xyxy']
        # save result
        np.save(os.path.join(npyfolder,basename+".npy"),historyresult)

def show_npy_label(
        imgfolder=r"Path/To/Folder"
    ):
    
    root=os.path.split(imgfolder)[0]
    npyfolder=os.path.join(root,'Detection')
    for file in tqdm(os.listdir(imgfolder)):
        basename,ext=file.split('.')
        imgpath=os.path.join(imgfolder,file)
        npypath=os.path.join(npyfolder,basename+'.npy')
        im0 = cv2.imread(imgpath)
        npydata=np.load(npypath,allow_pickle=True) # 'person': {'conf': 0.9263406, 'xyxy': [23.0, 335.0, 2026.0, 2048.0]}, 'helmet': {'conf': -1, 'xyxy': [614.0, 341.0, 1432.0, 1150.0]}}
        # get data by using npydata.item()[key]
        body_conf=npydata.item()['person']['conf']
        body_xyxy=npydata.item()['person']['xyxy']
        head_conf=npydata.item()['helmet']['conf']
        head_xyxy=npydata.item()['helmet']['xyxy']
        plot_one_box(body_xyxy, im0, label='body', color=[255,0,0], line_thickness=3)
        plot_one_box(head_xyxy, im0, label='head', color=[0,255,0], line_thickness=3)
        resized_img=cv2.resize(im0, dsize=(512 , 512), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("",resized_img)
        cv2.waitKey(5)

if __name__ == '__main__':
    with torch.no_grad():
        # detect_and_save()
        show_npy_label()
