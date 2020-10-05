import pickle
import re

import pandas as pd
import pydotplus
from IPython.display import Image, display
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from constants import feature_colname_subset_pattern


def tree_to_pseudo(tree, features_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [features_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    nodes = []

    def recurse(left, right, threshold, features, node, depth=0):
        indent = "  " * depth
        if (threshold[node] != -2):
            # print (indent,"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            nodes.append({"id": len(nodes),
                          "feature": features[node],
                          "threshold": threshold[node]})
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth + 1)
                # print (indent,"} else {")
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], depth + 1)
                # print (indent,"}")
        else:
            pass
            # print (indent,"return " + str(value[node]))

    recurse(left, right, threshold, features, 0)
    return nodes


TARGET_COLNAME = "TARGET"
PREDICTION_COLNAME = "prediction"
ACCURACY_COLNAME = "accuracy"


def add_agg_columns(df, colnames_to_aggregate, agg_type: str = "mean"):
    added_agg_colnames = []
    if agg_type == "mean":
        means_df = df[["1_char"] + [c for c in colnames_to_aggregate if "mean" not in c]] \
            .groupby(["1_char"]).mean()
        means_df.columns = [colname + '_' + agg_type for colname in means_df.columns]
        df = pd.merge(df, means_df, on=["1_char"])
        added_agg_colnames.extend(means_df.columns)

    # for column in colnames_to_aggregate:
    #     if pd.api.types.is_numeric_dtype(df[column])\
    #     and not pd.api.types.is_bool_dtype(df[column]):
    #         if agg_type == "mean":
    #             added_colname = f"{column}_agg_{agg_type}"
    #             df[added_colname] = df[column].mean()
    #             added_agg_colnames.append(added_colname)
    return df, added_agg_colnames


def add_tree_treshold_info(df, tree, tree_decisions_colnames):
    dt_nodes = tree_to_pseudo(tree, tree_decisions_colnames)
    for dt_node in dt_nodes:
        dt_node_id = dt_node["id"]
        dt_node_feature = dt_node["feature"]
        dt_node_treshold = dt_node["threshold"]
        dt_node_feature_colname = f"node_{dt_node_id}_feature"
        dt_node_treshold_colname = f"node_{dt_node_id}_threshold"
        dt_node_condition_colname = f"node_{dt_node_id}_condition"
        dt_node_divising_width_colname = f"node_{dt_node_id}_divising_width"

        df[dt_node_feature_colname] = dt_node_feature
        df[dt_node_treshold_colname] = dt_node_treshold
        dt_node_condition = df[dt_node_feature] <= dt_node_treshold
        df[dt_node_condition_colname] = dt_node_condition

        closest_lower_val = df.loc[df[dt_node_condition_colname] == True][dt_node_feature].max()
        closest_upper_val = df.loc[df[dt_node_condition_colname] == False][dt_node_feature].min()
        decision_width = closest_upper_val - closest_lower_val
        df[dt_node_divising_width_colname] = decision_width


def extract_save_balanced_dataset(dataset_fname,
                                  feature_colnames_pattern,
                                  feature_colname_subset_pattern,
                                  balanced_dataset_fname,
                                  agg_type="mean"):
    task_df = pd.read_csv(dataset_fname, delimiter='\t', encoding="utf-8")

    feature_colnames_candidates = [colname for colname in task_df.columns
                                   if re.match(feature_colnames_pattern, colname)]

    feature_colnames = [colname for colname in feature_colnames_candidates
                        if re.match(feature_colname_subset_pattern, colname)]

    task_df, added_agg_colnames = add_agg_columns(task_df, feature_colnames, agg_type)
    feature_colnames.extend(added_agg_colnames)

    # sample target class equisize
    task_df_g = task_df.groupby(TARGET_COLNAME)
    balanced_task_df = task_df_g.apply(lambda x: x.sample(task_df_g.size().min())) \
        .reset_index(drop=True)

    balanced_task_df.to_csv(balanced_dataset_fname, sep='\t', encoding="utf-8")

    return balanced_task_df, feature_colnames


def draw_tree(tree_dot_fn):
    with open(tree_dot_fn, 'r', encoding="utf-8") as tree_f:
        dot_data = tree_f.read()
    graph = pydotplus.graph_from_dot_data(dot_data)  # dot_data.getvalue())
    display(Image(graph.create_png()))


def dt_probe_dataset(task_fn, probing_hash, with_agg=True, agg_type="mean", tree=True):
    balanced_fname, prediction_fname = task_fn.replace('.tsv',
                                                       f'{probing_hash}_balanced.tsv'), f"{task_fn}.{probing_hash}.prediction"
    tree_dot_fn, tree_pkl_fn = task_fn.replace(".tsv", f"_{probing_hash}.dot"), task_fn.replace(".tsv",
                                                                                                f"_{probing_hash}.pkl")

    balanced_task_df, feature_colnames = extract_save_balanced_dataset(task_fn,
                                                                       r"(\d+_)?NN_FEAT",
                                                                       feature_colname_subset_pattern,
                                                                       balanced_fname,
                                                                       agg_type)

    if not with_agg:
        feature_colnames = [c for c in feature_colnames if agg_type not in c]

    features = balanced_task_df[feature_colnames]
    target = balanced_task_df[TARGET_COLNAME]

    if tree:
        dt = DecisionTreeClassifier(max_depth=2)
        dt.fit(features, target)
    else:
        dt = LogisticRegression(penalty="l1", solver="saga")
        dt.fit(features, target)
        logreg_feat_importance = list(zip(features.columns,
                                          dt.coef_[0]))

    accuracy = dt.score(features, target)
    balanced_task_df[ACCURACY_COLNAME] = accuracy
    balanced_task_df[PREDICTION_COLNAME] = dt.predict(features)

    if tree:
        add_tree_treshold_info(balanced_task_df, dt, feature_colnames)
        widths = [list(balanced_task_df[colname])[0]
                  for colname in balanced_task_df.columns if colname.endswith("divising_width")]
        # region save tree
        with open(tree_pkl_fn, 'wb') as f: pickle.dump(dt, f)
        export_graphviz(dt, out_file=tree_dot_fn, special_characters=True, feature_names=features.columns,
                        class_names=True, proportion=True, impurity=False)
        # endregion save tree

    balanced_task_df.to_csv(prediction_fname, sep='\t')

    res = {"balanced_fname": balanced_fname,
           "prediction_fname": prediction_fname,
           "tree_dot_fname": tree_dot_fn,
           "tree_pkl_fname": tree_pkl_fn,
           "accuracy": accuracy,
           "tree": tree,
           "features_importance": [] if tree else logreg_feat_importance,
           "widths": widths if tree else []}

    return res
