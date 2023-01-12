import numpy as np
import pandas as pd


### ----------------------- infer_columns.py ------------------------------

def infer_column_boundaries_from_clusters(layout_df, labels):
    assert type(labels) is np.ndarray
    assert type(layout_df) is pd.DataFrame
    
    labels = set(labels)
    column_boundaries = []
    
    gk = layout_df.groupby('col_label')

    for label in labels:
        x_1 = gk.get_group(label)['x_1'].describe()
        x_2 = gk.get_group(label)['x_2'].describe()
        
        # 1.5x std dev tolerance for min/max outliers, else use 25th or 75th percentile value respectively
        x_min = x_1.get('min') if x_1.get('min') > x_1.get('mean') - 1.5*x_1.get('std') else x_1.get('25%')
        x_max = x_2.get('max') if x_2.get('max') < x_2.get('mean') + 1.5*x_2.get('std') else x_2.get('75%')
        column_boundaries.append([label, x_min, x_max])
    
    return pd.DataFrame(columns=['col_label', 'x_min', 'x_max'], data=column_boundaries)


def infer_label_order_from_column_boundaries(column_boundaries_df):
    assert type(column_boundaries_df) is pd.DataFrame
    
    column_boundaries_df['x_bc'] = column_boundaries_df['x_min'] + 0.5*column_boundaries_df['x_max']
    column_boundaries_df.sort_values(by=['x_bc'], inplace=True)
    
    return column_boundaries_df['col_label'].to_list()
    

def error_flagger(row):
    if row['x_c'] < row['x_min']: return "MOVE_LEFT"
    if row['x_c'] > row['x_max']: return "MOVE_RIGHT"
    return "OK"


def relabel_flagged(row, **kwargs):
    labels = kwargs.get('labels')
    assert labels

    flag = row['flagged']
    label = row['col_label']

    if flag == "MOVE_RIGHT":
        row['labels']  = labels[labels.index(label)+1]
        row['flagged'] = "OK"
    if flag == "MOVE_LEFT":
        row['labels']  = labels[labels.index(label)-1]
        row['flagged'] = "OK"
    return row
