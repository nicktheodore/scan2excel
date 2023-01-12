import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from pdf2layout import main as pdf2layout

from utils import (
    infer_column_boundaries_from_clusters,
    infer_label_order_from_column_boundaries,
    error_flagger,
    relabel_flagged
)

def identify_column_clusters(layout_df, n_columns):
    assert type(layout_df) is pd.DataFrame
    assert type(n_columns) is int

    features = layout_df[['x_1', 'x_c', 'x_2']].to_numpy()

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_columns,
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    kmeans.fit(scaled_features)

    return kmeans


def main(layout_df, n_columns):
    assert type(layout_df) is pd.DataFrame
    assert type(n_columns) is int

    kmeans = identify_column_clusters(layout_df, n_columns)

    layout_df['labels']  = kmeans.labels_
    column_boundaries_df = infer_column_boundaries_from_clusters(layout_df, kmeans.labels_)
    labels               = infer_label_order_from_column_boundaries(column_boundaries_df)

    layout_df = pd.merge(layout_df, column_boundaries_df, how='left', on='labels')

    layout_df['flagged'] = layout_df.apply(error_flagger, axis=1)
    layout_df            = layout_df.apply(relabel_flagged, labels=labels, axis=1)

    return layout_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'infer_columns',
                    description = 'Infers column boundaries and classifies layout elements into column classes derived from k-means clusters',
                    epilog = 'Infers column boundaries and classifies layout elements into column classes derived from k-means clusters')

    parser.add_argument("path")
    parser.add_argument("n_cols")

    args = parser.parse_args()

    layout_df = pdf2layout(args.path)
    layout_df = main(layout_df, args.n_cols)

    print(layout_df)
