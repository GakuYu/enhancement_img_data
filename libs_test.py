
import os
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import PascalVocWriter
from scipy import misc

def loadPascalXMLByFilename(xmlPath):
    tVocParseReader = PascalVocReader(xmlPath)
    shapes = tVocParseReader.getShapes()
    return shapes

def savePascalVocFormat(filename, shapes, imagePath):
    imgFolderPath = os.path.dirname(imagePath)
    imgFolderName = os.path.split(imgFolderPath)[-1]
    imgFileName = os.path.basename(imagePath)
    #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
    # Read from file path because self.imageData might be empty if saving to
    # Pascal format

    image = misc.imread(imagePath, mode="RGB")
    [height, width, dim] = image.shape
    imageShape = [height, width, dim]
    writer = PascalVocWriter(imgFolderName, imgFileName,
                             imageShape, localImgPath=imagePath)
    writer.verified = False

    for shape in shapes:
        label = shape[0]
        box = shape[1]
        xmin = box[0][0]
        ymin = box[0][1]
        xmax = box[1][0]
        ymax = box[1][1]
        difficult = shape[2]
        writer.addBndBox(xmin, ymin, xmax, ymax, label, difficult)
    writer.save(targetFile=filename)
    return


if __name__ == '__main__':
    orig_annotation_folder = r'F:\公司资料\文档\AI竞赛\Surgical Instrument\外科工具\annotation'
    dst_annotation_folder = r'F:\公司资料\文档\AI竞赛\Surgical Instrument\外科工具\convert'

    for xml_file in os.listdir(orig_annotation_folder):
        [file_name,extension] =  xml_file.split(".")
        if xml_file.endswith('.xml'):
            shapes, image_name, image_path = loadPascalXMLByFilename(os.path.join(orig_annotation_folder, xml_file))
            savePascalVocFormat(os.path.join(dst_annotation_folder, xml_file), shapes, image_path)
    print("finish")