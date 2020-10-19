#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 图片数据增强
@Date: 2020/10/16 11:44:09
@Author: Gaku Yu
@version: 1.0
'''
import cv2 as cv
import numpy as np
from math import *

class ImageUtils:
    """
    @Description: 图片处理
    """

    @staticmethod
    def rotation(img, degree, boxes):
        """
        @Description: 图片旋转
        ---------
        @Args: 
            img: 原图片;
            degree: 旋转角度;
            boxes: 标注目标框数组, (xmin,ymin,xmax,ymax);
        -------
        @Returns: 
            dst: 旋转后的图片;
            dst_boxes: 旋转后的目标框数据组, (xmin,ymin,xmax,ymax);
        -------
        """
        
        
        ih, iw = img.shape[:2]

        v_sin = sin(radians(degree))
        v_cos = cos(radians(degree))

        nw, nh = int(iw*fabs(v_cos) + ih*fabs(v_sin)), int(iw*fabs(v_sin) + ih*fabs(v_cos))
        
        iw_half, ih_half, nw_half, nh_half = iw*0.5, ih*0.5, nw*0.5, nh*0.5

        mat_rotation = cv.getRotationMatrix2D((iw_half, ih_half), degree, 1)
        mat_rotation[0, 2] += (nw_half-iw_half)
        mat_rotation[1, 2] += (nh_half-ih_half)

        dst = cv.warpAffine(img, mat_rotation, (nw, nh), borderValue=(255,255,255))
        dst_boxes = list()
        
        for box in boxes:
            _x0, _x1, _y0, _y1 = (box[0]-iw_half), (box[2]-iw_half), (ih_half-box[1]), (ih_half-box[3])
            _x0_sin, _x0_cos, _x1_sin, _x1_cos = _x0*v_sin, _x0*v_cos, _x1*v_sin, _x1*v_cos
            _y0_sin, _y0_cos, _y1_sin, _y1_cos = _y0*v_sin, _y0*v_cos, _y1*v_sin, _y1*v_cos
            bx0, by0 = (_x0_cos - _y0_sin), (_x0_sin + _y0_cos)
            bx2, by2 = (_x1_cos - _y1_sin), (_x1_sin + _y1_cos)
            if degree % 90:
                bx1, by1 = (_x1_cos - _y0_sin), (_x1_sin + _y0_cos)
                bx3, by3 = (_x0_cos - _y1_sin), (_x0_sin + _y1_cos)
                bx, by = [bx0, bx1, bx2, bx3], [by0, by1, by2, by3]
                bx.sort()
                by.sort()
                dst_boxes.append([int(bx[0]+nw_half), int(nh_half-by[0]), int(bx[3]+nw_half), int(nh_half-by[3])])
            else:
                dst_boxes.append([int(bx0+nw_half), int(nh_half-by0), int(bx2+nw_half), int(nh_half-by2)])
        return dst, dst_boxes

    @staticmethod
    def flip(img, flip_type, boxes):
        """
        @Description: 镜像翻转图片
        ---------
        @Args: 
            img: 原图片;
            flip_type: 翻转类型, 1:水平, 0:垂直, -1:对角;
            boxes: 标注目标框数组, (xmin,ymin,xmax,ymax);
        -------
        @Returns: 
            dst: 旋转后的图片;
            dst_boxes: 旋转后的目标框数据组, (xmin,ymin,xmax,ymax);
        -------
        """


        ih, iw = img.shape[:2]
        if flip_type not in (1, 0, -1):
            raise Exception("Invalid flip_type, the correct value is 1, 0, or -1")
        dst = cv.flip(img, flip_type)
        dst_boxes = list()
        if flip_type == 1:
            for box in boxes:
                dst_boxes.append([iw-box[2], box[1], iw-box[0], box[3]])
        elif flip_type == 0:
            for box in boxes:
                dst_boxes.append([box[0], ih-box[3], box[2], ih-box[1]])
        else:
            for box in boxes:
                dst_boxes.append([iw-box[2], ih-box[3], iw-box[0], ih-box[1]])
        return dst, dst_boxes


import os
import sys
import logging as log
log.basicConfig(format="[ %(levelname)s | %(asctime)s ] %(message)s", level=log.INFO, stream=sys.stdout)
from xml.etree.ElementTree import ElementTree


class Enhancement:

    def __init__(self):
        self.img_dir = "images/"
        self.anno_dir = "annotation/"
        self.output_img_dir = "output/images/"
        self.output_anno_dir = "output/annotation/"
        pass

    def save_file(self, img, boxes, anno_tree, img_name, anno_name, label_name):
        ih, iw = img.shape[:2]
        img_folder = anno_tree.findtext("folder")
        img_dir = os.path.join(self.output_img_dir, img_folder)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        img_path = os.path.join(img_folder, img_name)
        anno_tree.find("filename").text = img_name
        anno_tree.find("path").text = img_path
        anno_size = anno_tree.find("size")
        anno_size.find("width").text = str(iw)
        anno_size.find("height").text = str(ih)

        for i, obj in enumerate(anno_tree.findall("object")):
            obj.find("name").text = label_name
            box_node = obj.find("bndbox")
            box_node.find("xmin").text = str(boxes[i][0])
            box_node.find("ymin").text = str(boxes[i][1])
            box_node.find("xmax").text = str(boxes[i][2])
            box_node.find("ymax").text = str(boxes[i][3])

        anno_tree.write(os.path.join(self.output_anno_dir, anno_name), encoding="utf-8")
        cv.imwrite(os.path.join(self.output_img_dir, img_path), img)

    def run(self):
        """
        @Description: 运行代码, 将原图片变换后保存到output文件夹
        ---------
        """
        if not os.path.exists(self.output_img_dir):
            os.mkdir(self.output_img_dir)
        if not os.path.exists(self.output_anno_dir):
            os.mkdir(self.output_anno_dir)
        for anno_file in os.listdir(self.anno_dir):
            try:
                anno_name, anno_ext =  anno_file.split(".")
                if anno_ext != "xml":
                    continue
                log.info(anno_file)
                label = "label_%s" % anno_name.split("_")[0]
                anno_tree = ElementTree()
                anno_tree.parse(os.path.join(self.anno_dir, anno_file))
                img_folder = anno_tree.findtext("folder")
                img_fullname = anno_tree.findtext("filename")
                img_name, img_ext =  img_fullname.split(".")
                img_path = os.path.join(self.img_dir, img_folder, img_fullname)
                
                src_img = cv.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
                src_box = list()
                for obj in anno_tree.iter(tag="object"):
                    box_node = obj.find("bndbox")
                    src_box.append(
                        [
                            int(box_node.findtext("xmin")),
                            int(box_node.findtext("ymin")),
                            int(box_node.findtext("xmax")),
                            int(box_node.findtext("ymax")),
                        ]
                    )
                self.save_file(src_img, src_box, anno_tree, 
                        "%s_0.%s" % (img_name, img_ext), "%s_0.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(src_img, 90, src_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_1.%s" % (img_name, img_ext), "%s_1.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(src_img, 180, src_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_2.%s" % (img_name, img_ext), "%s_2.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(src_img, 270, src_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_3.%s" % (img_name, img_ext), "%s_3.%s" % (anno_name, anno_ext), label)
                flip_img, flip_box = ImageUtils.flip(src_img, 1, src_box)
                self.save_file(flip_img, flip_box, anno_tree, 
                        "%s_4.%s" % (img_name, img_ext), "%s_4.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(flip_img, 90, flip_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_5.%s" % (img_name, img_ext), "%s_5.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(flip_img, 180, flip_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_6.%s" % (img_name, img_ext), "%s_6.%s" % (anno_name, anno_ext), label)
                dst_img, dst_box = ImageUtils.rotation(flip_img, 270, flip_box)
                self.save_file(dst_img, dst_box, anno_tree, 
                        "%s_7.%s" % (img_name, img_ext), "%s_7.%s" % (anno_name, anno_ext), label)
            except Exception as e:
                log.error(e)
        

def test_image():
    anno_dir = "output/annotation/"
    # for anno_file in os.listdir(anno_dir):
    for _i in range(8):
        anno_file = "%s_%s.xml" % ("1_0", _i)
        anno_tree = ElementTree()
        anno_tree.parse(os.path.join(anno_dir, anno_file))
        img = cv.imdecode(np.fromfile(os.path.join("output", "images", anno_tree.findtext("path")),dtype=np.uint8),-1)
        box = list()
        for obj in anno_tree.iter(tag="object"):
            box_node = obj.find("bndbox")
            box.append(
                [
                    int(box_node.findtext("xmin")),
                    int(box_node.findtext("ymin")),
                    int(box_node.findtext("xmax")),
                    int(box_node.findtext("ymax")),
                ]
            )
        for box in box:
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 4)
        cv.imshow("img", img)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == "__main__":
    enhancement = Enhancement()
    enhancement.run()
    # test_image()