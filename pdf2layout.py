import os
import argparse
import pdf2image
import numpy as np
import pandas as pd
import layoutparser as lp
from PIL.PpmImagePlugin import PpmImageFile


def convert_from_path(filepath):
    assert type(filepath) is str
    assert os.path.exists(filepath), "Error: {} does not exist".format(filepath)

    fid = os.path.basename(filepath)
    doc = pdf2image.convert_from_path(filepath)
    
    folder = 'doc'
    if folder not in os.listdir():
        os.makedirs(folder)

    for i, page in enumerate(doc):
        image_name = fid+"_page_"+str(i+1)+".jpg"  
        page.save(os.path.join(folder, image_name), "JPEG")

    return os.path.abspath(folder), doc


def detect_document_layout(img):
    assert type(img) is PpmImageFile

    # load pretrained Detectron2 model
    model = lp.Detectron2LayoutModel(
        "https://www.dropbox.com/s/h63n6nv51kfl923/config.yaml?dl=1",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0:"Table"})

    # predict document layout
    img = np.asarray(img)
    detected = model.detect(img)

    # sort and assign block ids
    new_detected = detected.sort(key=lambda x: x.coordinates[1])
    detected = lp.Layout([block.set(id=idx) for idx,block in 
                      enumerate(new_detected)])

    return detected, img
    

def extract_layout_contents(detected, img):
    assert type(detected) is lp.elements.layout.Layout
    assert type(img) is np.ndarray

    # load tesseract model
    ocr = lp.TesseractAgent(languages='eng+fra+ita')

    dic_predicted = {}
    for block in [block for block in detected if block.type in ["Table"]]:
        ## segmentation
        segmented = block.pad(left=15, right=15, top=5, 
                    bottom=5).crop_image(img)
        ## extraction
        extracted = ocr.detect(segmented,
                                agg_output_level=lp.TesseractFeatureType.LINE, 
                                return_response=True)
        
        text = extracted.get('text')
        
        ## save
        dic_predicted[str(block.id)+"-"+block.type] = text

    layout = ocr.gather_data(extracted, agg_level=lp.TesseractFeatureType.LINE)

    return layout


def create_layout_dataframe(layout):
    assert type(layout) is lp.elements.layout.Layout

    def get_x_center(row):
        return np.int64(row['x_1'] + (row['x_2']-row['x_1'])/2)

    def get_y_center(row):
        return np.int64(row['y_1'] + (row['y_2']-row['y_1'])/2)

    # convert layout to df
    layout_df = layout.to_dataframe()
    
    # drop empty bounding boxes
    layout_df['text'].replace(r'^\s*$', np.nan, inplace=True, regex=True)
    layout_df.dropna(subset=['text'], inplace=True)
    
    # get bounding box centers
    layout_df['x_c'] = layout_df.apply(get_x_center, axis=1)
    layout_df['y_c'] = layout_df.apply(get_y_center, axis=1)
    
    return layout_df


def main(filepath):
    assert type(filepath) is str

    layout_dfs = []
    _, doc        = convert_from_path(filepath)

    for img in doc: 
        detected, img = detect_document_layout(img)
        layout        = extract_layout_contents(detected, img)
        layout_df     = create_layout_dataframe(layout)
        layout_dfs.append(layout_df)

    layout_df = pd.concat(layout_dfs)
    
    return layout_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'pdf2layout',
                    description = 'Extracts text from tables in PDF scans using layout detection and OCR',
                    epilog = 'Extracts text from tables in PDF scans using layout detection and OCR')

    parser.add_argument("path")

    args = parser.parse_args()
    
    layout_df = main(args.path)
    print(layout_df)


