import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import os
import re

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from constants import feature_colname_subset_pattern


def get_row_viz_dict(row, nodes, node2viz_feature):
    def merge_dicts(dicts_sequence):
        res_di = {}
        for di in dicts_sequence:
            res_di.update(di)
        return res_di

    row_features_dicts = [node2viz_feature[node_idx][row[f"node_{node_idx}_condition"]] for node_idx in nodes]
    return merge_dicts(row_features_dicts)


def prettify_hidden_outs(hidden_outs_string):
    return re.sub(r"\d+_hidden_outs_(\d+)\1", r"hidden \g<1>", hidden_outs_string)


def prettify_dot(dot_fn):
    with open(dot_fn) as dot_f: dot_content = dot_f.read()
    dot_content_pretty = prettify_hidden_outs(dot_content)
    with open(dot_fn, 'w') as dot_f: dot_f.write(dot_content_pretty)


def add_tree_to_plot(tree_plot, tree_dot_path):
    tree_graph_fn = "tmp.txt"
    prettify_dot(tree_dot_path)
    os.system(f'dot -Tpng {tree_dot_path} -o {tree_graph_fn} -Gdpi=500 -Gratio=1.3')
    tree_plot.imshow(mpimg.imread(tree_graph_fn))


def draw_examples(nodes_ixes, df, nodes_plots, colname2bounds):
    x_col = "1_char" if "1_char" in df.columns else "0_ngram"
    x_seq = df[x_col]

    default_point_size = plt.rcParams['lines.markersize'] ** 2

    for node_idx in nodes_ixes:

        y_col = f'node_{node_idx}_feature_value'
        y_seq = df[y_col]

        target = df['TARGET']
        prediction = df['prediction']
        mpl_params = df['mpl_params']
        for x, y, tgt, pred, features_di in zip(x_seq, y_seq, target, prediction, mpl_params):
            color = features_di["color"]
            fill = features_di.get("fill", color)
            if fill == 'full':
                fill = color

            shape = features_di.get("marker", 's')

            nodes_plots[node_idx].scatter(x, y, facecolors=fill, edgecolors=color, marker=shape)

            if tgt != pred:
                if shape == "s":
                    ershape = '_'
                if shape == "d":
                    ershape = '|'
                nodes_plots[node_idx].scatter(x, y, marker=ershape, c='r')

        decision_bound_col = f"node_{node_idx}_threshold"
        decision_bound = list(set(df[decision_bound_col]))[0]
        nodes_plots[node_idx].axhline(y=decision_bound, color='grey', linestyle=':')

        y_mean = df[[x_col, y_col]].groupby(x_col, as_index=False).mean()
        for x, y in zip(y_mean[x_col], y_mean[y_col]):
            nodes_plots[node_idx].scatter(x, y, marker="_", c='black', s=default_point_size * 2)

        ymin = colname2bounds[y_col]["ymin"]
        ymax = colname2bounds[y_col]["ymax"]
        nodes_plots[node_idx].set_ylim(ymin, ymax)


def get_distinctive_viz_features():
    distinctive_viz_feature_1 = {True: {"color": "y"}, False: {"color": "b"}}
    distinctive_viz_feature_2 = {True: {"fill": "full"}, False: {"fill": "none"}}
    distinctive_viz_feature_3 = {True: {"marker": "s"}, False: {"marker": "d"}}
    return [distinctive_viz_feature_1,
            distinctive_viz_feature_2,
            distinctive_viz_feature_3]


def get_interesting_columns(df):
    interesting_columns = [colname
                           for colname in df.columns
                           if ("_char_" not in colname and "_word_" not in colname)
                           and ("word" in colname or "char" in colname or "ngram" in colname
                                or re.match(r"node_\d+_(condition|threshold|feature|feature_value)", colname)
                                or colname in ("accuracy", "prediction", "TARGET")
                                or re.match(feature_colname_subset_pattern, colname))]  # + feature_columns
    return interesting_columns


def get_fecture_colname_col2feature_colname(df):
    _feature_columns = [col_name
                        for col_name in df.columns
                        if re.match(r"node_\d+_feature", col_name)]
    feature_columns = {c: f_name for c in df for f_name in df[c] if c in _feature_columns}
    return feature_columns


def foo(log):
    data_path = log["prediction_fname"]
    dot_path = log["tree_dot_fname"]  # [data_path.replace(".tsv", ".dot")
    vvc_df = pd.read_csv(data_path, sep='\t')
    tree_graph_fn = "file.png"

    feature_columns = get_fecture_colname_col2feature_colname(vvc_df)

    for feature_column_column, feature_column in feature_columns.items():
        vvc_df[feature_column_column + "_value"] = vvc_df[feature_column]

    interesting_columns = get_interesting_columns(vvc_df)
    print(interesting_columns)

    nodes_ixes = {int(col_name.split('_')[1])
                  for col_name in interesting_columns
                  if re.match(r"node_\d+_feature_value", col_name)}
    distinctive_viz_features = get_distinctive_viz_features()
    node2viz_feature = dict(zip(nodes_ixes, distinctive_viz_features))

    vvc_df_small = vvc_df[sorted(interesting_columns)]
    vvc_df_small["mpl_params"] = vvc_df_small.apply(lambda x: get_row_viz_dict(x, nodes_ixes, node2viz_feature), axis=1)

    plt.rcParams['figure.figsize'] = [15, 5]

    fig = plt.figure(figsize=(20, 15), frameon=True)

    figure_gridspec = gridspec.GridSpec(1, 2, figure=fig)
    activations_gridspec = gridspec.GridSpecFromSubplotSpec(len(nodes_ixes), 2, subplot_spec=figure_gridspec[0])

    tree_plot = fig.add_subplot(figure_gridspec[:, 1:])  # all rows, all but first cols. i.e the right part of plot

    tree_plot.set_title("Tree plot")

    add_tree_to_plot(tree_plot, dot_path)

    feature_names = []
    for node_ix in nodes_ixes:
        feature_name = list(set(vvc_df_small[f"node_{node_ix}_feature"]))[0]
        feature_names.append(prettify_hidden_outs(feature_name))

    nodes_true_plots = [fig.add_subplot(activations_gridspec[node_idx, 0]) for node_idx in nodes_ixes]
    nodes_true_plots[0].set_title("target = True")
    for node_idx in nodes_ixes:
        nodes_true_plots[node_idx].set_ylabel(feature_names[node_idx], size='large')

    nodes_false_plots = [fig.add_subplot(activations_gridspec[node_idx, 1]) for node_idx in nodes_ixes]
    nodes_false_plots[0].set_title("target = False")

    colname2bounds = {}
    for node_idx in nodes_ixes:
        colname = f"node_{node_idx}_feature_value"
        ymax = np.ceil(vvc_df_small[colname].max())
        ymin = np.floor(vvc_df_small[colname].min())
        bounds = {"ymin": ymin, "ymax": ymax}
        colname2bounds[colname] = bounds

    draw_examples(nodes_ixes, vvc_df_small.loc[vvc_df_small.TARGET == True], nodes_true_plots, colname2bounds)
    draw_examples(nodes_ixes, vvc_df_small.loc[vvc_df_small.TARGET == False], nodes_false_plots, colname2bounds)

    return fig
