from xml.etree.ElementTree import Element, ElementTree, tostring
import json, csv

csvfile = './VOC2007/Annotations/part3.csv'   #TODO
xmlfile = './VOC2007/Annotations/'
jpgpath = './VOC2007/JPEGImages/'

'''
JPG means the name of JPG
'''

def csvtoxml(csvname, JPG):
    with open(csvname, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        header = ['name', 'region_id','bndbox']
        root = Element('Annotation')
        filename = Element('filename')   #path
        filename.text = JPG + '.jpg'
        root.append(filename)
        path = Element('path')
        path.text = jpgpath + JPG + '.jpg'
        root.append(path)
        key = 0

        for row in reader:
            if row[0].startswith(JPG):  #judge the filename
                print(row)
                if key == 0:
                    key = 1
                    bike_count = Element('bike_count')
                    bike_count.text = row[2].split("\"")[3]
                    root.append(bike_count)

                #add object

                object = Element('object')  #element
                xy = ['xmin','ymin','xmax','ymax']
                name = Element('name')
                target = row[5]
                name.text = row[6].split("\"")[3]
                object.append(name)
                bndbox = Element('bndbox')
                tar = target.split(":")
                print(tar)
                x, y = int(tar[2].split(",")[0]), int(tar[3].split(",")[0])
                xmax, ymax = int(tar[4].split(",")[0]), int(tar[5][:-1])
                for ele, value in zip(xy, [x, y, x+xmax, y+ymax]):
                    e = Element(ele)
                    e.text = str(value)
                    bndbox.append(e)
                    print(ele, value)

                object.append(bndbox)
                root.append(object)

    beatau(root)
    return ElementTree(root)


def beatau(e, level=0):
    if len(e) > 0:
        e.text = '\n' + '\t' * (level + 1)
        for child in e:
            beatau(child, level + 1)
        child.tail = child.tail[:-1]
    e.tail = '\n' + '\t' * level

if __name__ == '__main__':
    JPG = ['tagbike{}'.format(i) for i in range(200,300)]   #TODO

    for jpg in JPG:
        et = csvtoxml(csvfile, jpg)
        et.write(xmlfile+jpg+'.xml')