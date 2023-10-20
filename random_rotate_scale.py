# random rotate/scale dataset
# using cv2.getAffineTransform & cv2.warpAffine to align a random rotated and scaled region(represented as sympy.RegularPolygon) from a 2048x2048pixel img.
# pip install sympy
import cv2
import numpy as np
from tqdm import tqdm
from time import time
# pip install Sympy , online help: https://www.osgeo.cn/sympy/modules/geometry/index.html
# from sympy import *
from sympy import Point2D,Polygon,Line,Circle,Segment2D,pi,RegularPolygon
import os
srcroot=r"D:/NN/Daz3D_Human_dataset"
dstroot=r"D:/dataset/DAZ3d_rotate_crop"
ROI_init_rotation=pi/4
ROI_rect_rotation_range=[-60/180*pi,60/180*pi]
randomrange=ROI_rect_rotation_range[1]-ROI_rect_rotation_range[0]
randomstart=ROI_rect_rotation_range[0]+ROI_init_rotation
IMG_size=2048
multiply_=2 

def parse_detection_npydata(file=r"path/smart_construction_detection_example.npy"):
    npdata=np.load(file=file,allow_pickle=True)
    try:
        body_x0,body_y0,body_x1,body_y1=npdata.item()['person']['xyxy']
    except:
        body_x0=body_y0=body_x1=body_y1=IMG_size/2
    try:
        head_x0,head_y0,head_x1,head_y1=npdata.item()['helmet']['xyxy']
    except:
        head_x0=head_y0=head_x1=head_y1=IMG_size/2
    # in NN output, x means w. well in Sympy, y means w (its a 90 CW-rotated cord-world)
    head_center_wy=(head_x0+head_x1)/2 
    head_center_hx=(head_y0+head_y1)/2 
    head_radius=np.absolute((head_y1-head_y0)/2*1.5) # scale up a little, in case face is cropped
    body_center_wy=(body_x0+body_x1)/2 
    body_center_hx=(body_y0+body_y1)/2 
    # body_radius=np.max(
    #     np.absolute((body_y1-body_y0)/2),
    #     np.absolute((body_x1-body_x0)/2),
    # )
    headcenter=Point2D(head_center_hx,head_center_wy)
    bodycenter=Point2D(body_center_hx,body_center_wy)
    headcenter_to_bodycenter_segment=Segment2D(headcenter,bodycenter)
    return headcenter_to_bodycenter_segment, headcenter, head_radius # sympy format

def create_random_rotate_scale_img_dataset(IPfolder):
    src_IPfolder=os.path.join(srcroot,IPfolder)
    detection_folder=os.path.join(src_IPfolder,"Detection")
    for subfolder in os.listdir(src_IPfolder):
        subfolderpath=os.path.join(src_IPfolder,subfolder)
        if os.path.isdir(subfolderpath) and subfolder!="Detection":
            print('     ==>',subfolderpath)
            dstfolder=os.path.join(dstroot,IPfolder,subfolder)
            os.makedirs(dstfolder,exist_ok=True)
            for file in tqdm(os.listdir(subfolderpath)):
                filepath=os.path.join(subfolderpath,file)
                basename,ext=file.split('.')
                detectionfile=os.path.join(detection_folder,basename+".npy")
                counter=0
                img=cv2.imread(filepath)
                assert img.shape==(2048,2048,3)
                headcenter_to_bodycenter_segment, headcenter, head_radius=parse_detection_npydata(detectionfile)
                # create data
                for i in range(multiply_):
                    # random ROI params
                    ROI_centerpoint=headcenter_to_bodycenter_segment.random_point(seed=None) 
                    ROI_radius=headcenter.distance(ROI_centerpoint)+head_radius
                    ROI_rectangle_raduis=ROI_radius*1.414
                    ROI_rotation=randomrange*np.random.random()+randomstart
                    # create a square，vertex:[(r,0),(0,r),(-r,0)(0,-r)] from Sympy POV. Well from IMG POV, the 1st vertex is the lowest point in the img.
                    ROI_rectangle=RegularPolygon(
                        c=ROI_centerpoint, # center point
                        r=ROI_rectangle_raduis, # radius
                        n=4, # edges
                        rot=ROI_rotation # rotation
                        )
                    # print(ROI_rectangle.vertices[0]) # (hx,wy)
                    # vertex sequence: v[2],v[3],v[0]
                    origin=[ROI_rectangle.vertices[2].y,ROI_rectangle.vertices[2].x] # cordinate order:(w,h), aligh with final img (0,0)
                    lower_left_corner=[ROI_rectangle.vertices[3].y,ROI_rectangle.vertices[3].x] # cordinate order:(w,h), aligh with final img (0,1023)
                    lower_right_corner=[ROI_rectangle.vertices[0].y,ROI_rectangle.vertices[0].x] # cordinate order:(w,h), aligh with final img (1023,1023)
                    # like AutoCAD 'align' command, 3 groups of paired vertex needed.
                    pt1 = np.float32([origin, lower_left_corner, lower_right_corner]) 
                    pt2 = np.float32([[0, 0], [0,1023], [1023, 1023]]) # origin, lower_left_corner, lower_right_corner's cords in the final img
                    M = cv2.getAffineTransform(pt1, pt2)
                    result= cv2.warpAffine(img, M, dsize=(1024,1024))
                    # save
                    dstfile=os.path.join(dstfolder,basename+"_"+str(counter).zfill(2)+".jpg")
                    counter+=1
                    cv2.imwrite(filename=dstfile,img=result,params=[int(cv2.IMWRITE_JPEG_QUALITY),100])

if __name__=='__main__':
    create_random_rotate_scale_img_dataset()
