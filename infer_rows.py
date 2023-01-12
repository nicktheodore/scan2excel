import json
import argparse
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import  DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


def identify_row_clusters(layout_df, n_columns, min_samples=1):
    assert type(layout_df) is pd.DataFrame
    assert type(n_columns) is int

    # Feature selection for column and row clusters, respectively
    y_features = layout_df[['y_1', 'y_c', 'y_2']].to_numpy()

    # Feature scaling for column and row data, respectively
    y_scaler = RobustScaler()
    y_scaled_features = y_scaler.fit_transform(y_features)

    # KNN analysis for row data
    nbrs = NearestNeighbors(n_neighbors=n_columns-1).fit(y_scaled_features)
    neigh_dist, _   = nbrs.kneighbors(y_scaled_features)
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    # Isolate furthest neighbors
    k_dist = sort_neigh_dist[:, -1]

    # Find epsilon value for DBSCAN @ knee of distance curve
    kl  = KneeLocator(range(0, len(k_dist)), k_dist, curve="convex", direction="increasing")
    eps = kl.elbow_y

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(y_scaled_features)

    return dbscan


def process_row_labels(layout_df, dbscan, column_labels):
    assert type(layout_df) is pd.DataFrame
    assert type(dbscan) is DBSCAN

    layout_df['row_label'] = dbscan.labels_

    y_median = layout_df.groupby('row_label').agg(y_m=('y_c', np.median))
    layout_df = pd.merge(layout_df, y_median, how='left', on='row_label')

    layout_df = layout_df.sort_values('y_m', ascending=True).reset_index()
    grouped_rows = layout_df.groupby('row_label', sort=False)

    rows = []
    for _, group in grouped_rows:
        row = group.set_index('col_label').T.loc['text'].to_dict()
        rows.append(row)
        
    table_df = pd.DataFrame(rows)[column_labels]

    return table_df


def main(layout_df, column_labels, **kwargs):
    assert type(layout_df) is pd.DataFrame
    assert type(column_labels) is list

    dbscan = identify_row_clusters(layout_df, len(column_labels), **kwargs)
    table_df = process_row_labels(layout_df, dbscan, column_labels)

    return table_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'infer_rows',
                    description = 'Infers row groupings and classifies layout elements into row classes derived from DBSCAN clusters',
                    epilog = 'Infers row groupings and classifies layout elements into row classes derived from DBSCAN clusters')
    
    layout_df = pd.read_csv('data/infer_columns.csv')
    
    with open('data/ordered_labels', 'r') as f: column_labels = json.load(f)

    layout_df = main(layout_df, column_labels)
    layout_df.to_csv('data/infer_rows.csv')

    print(layout_df)
    