import tensorflow as tf
import cv2
import os
import glob
import time
import pdf2image
import pytesseract as pt
import codecs
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
from utils import visualization_utils as vis_util
from googletrans import Translator
from tkinter import Tk
from tkinter.filedialog import askdirectory
import re
import pandas as pd

# To get Folder Path------------------------------------------------------------------------------------------------------------
def browse_file_path():
    # gui to take input from user----------------------
    Tk().withdraw()
    dir_path = askdirectory()
    return dir_path

# Convert Text To Excel---------------------------------------------------------------------------------------------------------
def text_to_Excel(inputPath, outputPath):
    ''' From, Here you add your code'''
    print(inputPath, outputPath)
    folder = glob.glob(inputPath + "/*")
    import codecs
    for text_file in folder:
        print(text_file)
        f = codecs.open(text_file, encoding="utf-8", mode="r")
        translated_text = f.read()
        m = re.findall(r"नाम :\s+(.*)", translated_text)
        n = re.findall(r"का नाम:\s+(.*)", translated_text)
        o1 = re.findall(r"मकान क्रमांक:\s.*?(?=\s)+", translated_text)
        o2 = re.findall(r"मकानक्रमांक:\s.*?(?=\s)+", translated_text)
        p1 = re.findall(r"आयु :\s.*?(?=\s)+", translated_text)
        p2 = re.findall(r"आयु:\s.*?(?=\s)+", translated_text)
        q = re.findall(r"लिंग:\s.*?(?=\s)", translated_text)
        # [\d—-]+
    
        m = m[0] if m else ""
        # n1 = n1[0] if n1 else ""
        n = n[0] if n else ""
        o1 = o1[0] if o1 else ""
        o2 = o2[0] if o2 else ""
        # o3 = o3[0] if o3 else ""
        # o4 = o4[0] if o4 else ""
        p1 = p1[0] if p1 else ""
        p2 = p2[0] if p2 else ""
        q = q[0] if q else ""

        # line = m + ',' + n1  + n2 + ',' + o1  + o2  + o3  + o4 + ',' + p + ',' + q
        line = m + "," + n + "," + o1 + o2 + "," + p1 + p2 + "," + q
        # file.write(line)
        # del_list = ["Name:", " Name:", "name:", "Number:", "number:", "No:", "no:", "Age:", "Sex:"]
        del_list = ["नाम :", "का नाम:", "मकान क्रमांक:", "आयु :", "लिंग:", "आयु:", "मकानक्रमांक:", "१", "।"]

        line1 = line.replace(del_list[0], "")
        line2 = line1.replace(del_list[1], "")
        line3 = line2.replace(del_list[2], "")
        line4 = line3.replace(del_list[3], "")
        line5 = line4.replace(del_list[4], "")
        line6 = line5.replace(del_list[5], "")
        line7 = line6.replace(del_list[6], "")
        line8 = line7.replace(del_list[7], "1")
        line9 = line8.replace(del_list[8], "1")




        # saving the  text for every image in a separate .txt file
        
        import codecs
        file1 = codecs.open(outputPath + '/outputFile.txt', encoding='utf-8', mode="a+")
        file1.write(line9   +"\n")
        file1.close()
        
        df = pd.read_csv(outputPath + "/outputFile.txt", sep = ",", names = ["नाम", "पति/पिता का नाम", "मकान क्रमांक", "आयु", "लिंग"])
        # df.to_csv("D:\\sps_documents\\python\\tesseract_OD_pdf\\page_crop\\final_txt_csv\\output.csv")
        # df_new = pd.read_csv("D:\\sps_documents\\python\\tesseract_OD_pdf\\page_crop\\final_txt_csv\\output.csv") 
  
        # saving xlsx file 
        GFG = pd.ExcelWriter(outputPath + '/output_excel.xlsx') 
        df.to_excel(GFG, index = False) 
          
        GFG.save()
    

def pdf_to_text(detection_graph, category_index):
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # # PDF Folder Path------------------------------------------------------------------------------------------------------------
    # print("\t| Select Input Dir/Folder:-------")
    # PATH_TO_PDF_FOLDER = browse_file_path()
    PATH_TO_PDF_FOLDER = "D:\\sps_documents\\python\\tesseract_OD_pdf\\page_crop\\pdf_dir"

    # time.sleep(2)

    # Output Text_files----------------------------------------------------------------------------------------------------------
    # print("\t| Select Text File Dir/Folder:---")
    # PATH_TO_TEXT = browse_file_path()
    PATH_TO_TEXT = "D:\\sps_documents\\python\\tesseract_OD_pdf\\page_crop\\text_files"
    # time.sleep(2)

    # Output Excel file dir-------------------------------------------------------------------------------------------------------
    # print("\t| Select Excel File Dir/Folder:--\n")
    # PATH_TO_DF = browse_file_path()
    PATH_TO_DF = "D:\\sps_documents\\python\\tesseract_OD_pdf\\page_crop\\final_txt_csv"

    time.sleep(2)

    print("\t| Input Dir:-      ",PATH_TO_PDF_FOLDER,"\n\t| Text File Dir:-  ",PATH_TO_TEXT,"\n\t| Excel File Dir:- ",PATH_TO_DF,'\n\n')

    # MAIN EXECUTION-------------------------------------------------------------------------------------------------------------
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph-----------------------------------------------------------
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # For Each pdf One by One-----------------------------------------------------------------------------------------------------
        PDF_FOLDER = glob.glob(PATH_TO_PDF_FOLDER + "/*")

        for pdf in PDF_FOLDER:
            
            pdf_str = pdf.replace('\\','\\\\')
            pdf_str_list = pdf_str.split('\\\\')

            pdf_name = pdf_str_list[-1][:-4]
            # print(pdf_name)

            index = 1 # For Pdf Img
            count = 1 # For Crop Img
            
            # To check processing time each pdf--------------------------
            now = time.perf_counter()

            # DECLARE CONSTANTS-------------------------------------------------------------------------------------------------------
            # PDF_PATH = filename
            PDF_PATH = pdf
            DPI = 200
            OUTPUT_FOLDER = None
            FIRST_PAGE = 3
            LAST_PAGE = None
            FORMAT = 'jpg'
            THREAD_COUNT = 1
            USERPWD = None
            USE_CROPBOX = False
            STRICT = False
            print("\n\n\t|--------------WORKING ON PDF:- '{}'\n".format(pdf_str_list[-1]))

            # Convert pdf to images...------------------------------------------------------------------------------------------------
            pil_images = pdf2image.convert_from_path(PDF_PATH, dpi=DPI, output_folder=OUTPUT_FOLDER, first_page=FIRST_PAGE,
                                                     last_page=LAST_PAGE, fmt=FORMAT, thread_count=THREAD_COUNT, userpw=USERPWD,
                                                     use_cropbox=USE_CROPBOX, strict=STRICT)


            for image in pil_images:
                # Convert Pil Image to CV2 image---------------------------------------------------------------------------------------
                cv2_image = np.array(image)
                image_expanded = np.expand_dims(cv2_image, axis=0)

                # Perform the actual detection by running the model with the image as input---------------------------------------------
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})


                # Draw the results of the detection (aka 'visulaize the results')-------------------------------------------------------

                vis_util.visualize_boxes_and_labels_on_image_array(
                    cv2_image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.80)


                # Crop the Detected Part from image..--------------------------------------------------------------------------------------
                (frame_height, frame_width) = cv2_image.shape[:2]
                for i in range(len(np.squeeze(scores))):
                            
                    # print(np.squeeze(boxes)[i])
                    # print(np.squeeze(scores)[i])
                    if np.squeeze(scores)[i] > 0.8:
                        ymin = int((np.squeeze(boxes)[i][0]*frame_height))
                        xmin = int((np.squeeze(boxes)[i][1]*frame_width))
                        ymax = int((np.squeeze(boxes)[i][2]*frame_height))
                        xmax = int((np.squeeze(boxes)[i][3]*frame_width))

                        # Crop Form Pdf Pilimage--------------------------------------------
                        cropped_img = image.crop((xmin, ymin, xmax, ymax))

                        text = pt.image_to_string(cropped_img, lang="hin", config='--psm 6')
                        # text.replace("।", "1")
                        # text = tesserocr.image_to_text(cropped_img, lang="hin")

                        # saving the  text for every image in a separate .txt file-------------------------------------------------------
                        f = codecs.open(PATH_TO_TEXT + "/{}_crop_{}.txt".format(pdf_name, count), encoding='utf-8', mode="w")
                        f.write(text)
                        f.close()

                        count +=1
                    else:
                        print("\t|----ALL OK FOR PDF:- '{}' PAGE_No:- {}".format(pdf_str_list[-1],index))
                        break
                        # End of 2nd for loop------------

                # Pdf page index------------------
                index += 1

            # Timer-------------------------------------------
            to_print = "\n\t|--------------PDF:- {} will take ".format(pdf_str_list[-1])
            print(to_print + str(int(time.perf_counter() - now)) + " Seconds.\n\n")
            # End of 1st for loop-----------------------------------

    # Convert Text Files to Excel----------------------------------------------------------------------------------------------------------
    text_to_Excel(PATH_TO_TEXT, PATH_TO_DF)

    # Last Execution-----------------------------------------------------------------------------------------------------------------------
    print("\t|...............................................Completed...............................................|")