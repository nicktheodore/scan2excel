import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from infer_columns import main as infer_columns
from infer_rows import main as infer_rows

PROJECT_PATH = ''

sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)


parser = argparse.ArgumentParser(
                    prog = 'scan2excel',
                    description = 'Extracts text from tables in PDF scans and converts them to Excel spreadsheets',
                    epilog = 'Extracts text from tables in PDF scans and converts them to Excel spreadsheets')


parser.add_argument('infile', type=str)
parser.add_argument("col_names", action="extend", nargs="+", type=str)
parser.add_argument('-o', '--outpath', default='tables/', type=str)
if not os.path.exists('tables/'): os.mkdir('tables/')
    
args = parser.parse_args()

# main.py will run pdf2layout.py separately due to segfault bug in layoutparser:
# https://github.com/Layout-Parser/layout-parser/issues/13

#os.system('conda activate opencv')
os.system('python pdf2layout.py {}'.format(args.infile))

layout_df = pd.read_csv('data/pdf2layout.csv')
layout_df, column_labels = infer_columns(layout_df, len(args.col_names))
layout_df                = infer_rows(layout_df, column_labels)


filepath = os.path.join(args.outpath, 
                    os.path.splitext(os.path.basename(args.infile))[0] + '-' + \
                        str(datetime.now().timestamp()).split('.',1)[0]
                    )

layout_df.rename(columns=dict(zip(column_labels, args.col_names)), inplace=True)
layout_df.to_excel(filepath+'.xlsx')
print(layout_df)
