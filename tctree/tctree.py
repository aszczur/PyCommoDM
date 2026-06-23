import math
from collections import Counter
from enum import Enum
import numpy as np

class TemporalCutsQualityMeasure(Enum):
    DISC_PAIR_FROM_DIFF_CLASSES = 1
    DISC_PAIR_FROM_THE_SAME_CLASS = 2
    D_PAIR_DIFF_CLASSES_BY_D_PAIR_THE_SAME_CLASS = 3

class BestTemporalCutSelectionMethod(Enum):
    MAX_QUALITY = 1
    MAX_QUALITY_MAX_ALPHA = 2
    MAX_QUALITY_MIN_ALPHA = 3

class TemporalCutsTreeNode:
    def __init__(self, dominant_class_ratio, num_objects, num_objects_per_class, decision_class):
        self.dominant_class_ratio = dominant_class_ratio
        self.num_objects = num_objects
        self.num_objects_per_class = num_objects_per_class
        self.decision_class = decision_class
        self.attribute_name = None
        self.attribute_index = None
        self.attribute_value = None
        self.is_up_cut = None
        self.alpha_of_time_points = None
        self.quality = None
        self.ids_obj_over_cut = None
        self.left = None
        self.right = None

class TemporalCutsDecisionTreeClassifier:
    def __init__(self, max_depth=None, pruning_factor=1.0):
        self.alpha_param_start = None
        self.alpha_step = None
        self.alpha_step_count = None
        self.quality_measure = None
        self.quality_sel_method = None
        self.tree_ = None
        self.attributes_names = None
        self.time_points = None
        self.object_ids = None
        self.max_depth = max_depth
        self.pruning_factor = pruning_factor
        self.classes_ = None
        self.n_classes_ = None
        self.n_attributes_ = None


    def predict(self, data_tst):
        id_col = data_tst.columns[0]
        predicted = []
        for tw_id, df_tw in data_tst.groupby(id_col, sort=False):
            predicted.append(self._predict_single_tw(df_tw))
        return predicted


    def _predict_single_tw(self, window_tst):
        no_column = window_tst.shape[1]
        inputs = np.array(window_tst.iloc[:, 2:no_column - 1])
        tp_count = inputs.shape[0]

        node = self.tree_
        while node.left:
            cut_column = inputs[:, node.attribute_index]
            if node.is_up_cut:
                tp_over_cut_count = np.sum(cut_column >= node.attribute_value)
            else:
                tp_over_cut_count = np.sum(cut_column < node.attribute_value)
            tp_over_cut_ratio = tp_over_cut_count / tp_count

            if tp_over_cut_ratio >= node.alpha_of_time_points:
                node = node.left
            else:
                node = node.right
        return node.decision_class


    def predict_proba(self, data_tst):
        id_col = data_tst.columns[0]
        data_tst[id_col] = data_tst[id_col].astype(str)
        probs = []
        for tw_id, df_tw in data_tst.groupby(id_col, sort=False):
            probs.append(self._predict_proba_single_tw(df_tw))
        return probs


    def _predict_proba_single_tw(self, window_tst):
        no_column = window_tst.shape[1]
        inputs = np.array(window_tst.iloc[:, 2:no_column - 1])
        tp_count = inputs.shape[0]

        node = self.tree_
        while node.left:
            cut_column = inputs[:, node.attribute_index]
            if node.is_up_cut:
                tp_over_cut_count = np.sum(cut_column >= node.attribute_value)
            else:
                tp_over_cut_count = np.sum(cut_column < node.attribute_value)
            tp_over_cut_ratio = tp_over_cut_count / tp_count

            if tp_over_cut_ratio >= node.alpha_of_time_points:
                node = node.left
            else:
                node = node.right

        counts = node.num_objects_per_class
        total = sum(counts)
        if total == 0:
            probs = [0.0 for _ in counts]
        else:
            probs = [c/total for c in counts]
        return probs


    def fit(self, data_tr, alpha_start=1.0, alpha_step=0, alpha_step_count=0,
            quality_measure=TemporalCutsQualityMeasure.DISC_PAIR_FROM_DIFF_CLASSES,
            quality_sel_method = BestTemporalCutSelectionMethod.MAX_QUALITY):
        no_column = data_tr.shape[1]
        features_tr = np.array(data_tr.iloc[:, 2:no_column - 1])
        labels_tr = np.array(data_tr.iloc[:, no_column - 1])
        self.classes_ = np.unique(labels_tr)
        self.n_classes_ = len(self.classes_)
        self.n_attributes_ = features_tr.shape[1]
        id_tw_column = 0
        id_tp_column = 1
        self.object_ids = np.array(data_tr.iloc[:, id_tw_column])
        self.time_points = np.array((data_tr.iloc[:, id_tp_column]))
        col_names_tr = list(data_tr.columns)
        self.attributes_names = col_names_tr[2:-1]
        self.alpha_param_start = alpha_start
        self.alpha_step = alpha_step
        self.alpha_step_count = alpha_step_count
        self.quality_measure = quality_measure
        self.quality_sel_method = quality_sel_method
        self.tree_ = self._grow_tree(features_tr, labels_tr, self.object_ids, self.time_points, self.attributes_names,
                                     self.alpha_param_start, self.alpha_step, self.alpha_step_count,
                                     self.quality_measure, self.quality_sel_method)


    def _num_obj_per_class(self, ids, y):
        combined = np.vstack((ids, y))
        res = {}
        for v_y in self.classes_:
            uniq = np.unique(combined[0][combined[1] == v_y])
            res[v_y] = len(uniq)
        return res


    def _dominant_class_ratio(self, ids, y):
        num_obj_per_class = self._num_obj_per_class(ids, y)
        maximum = max(num_obj_per_class.values())
        count = sum(num_obj_per_class.values())
        return maximum/count


    def _dominant_class_value(self, ids, y):
        num_obj_per_class = self._num_obj_per_class(ids, y)
        max_key = max(num_obj_per_class, key=num_obj_per_class.get)
        return float(max_key)


    def _grow_tree(self, X, y, object_ids, time_points, attributes_names, alpha_param_start, alpha_step,
                   alpha_step_count, quality_measure, quality_sel_method, depth=0):
        num_objects_per_class = list(self._num_obj_per_class(object_ids, y).values())
        dominant_class_ratio = self._dominant_class_ratio(object_ids, y)
        dominant_class_value = self._dominant_class_value(object_ids, y)

        node = TemporalCutsTreeNode(
            dominant_class_ratio = dominant_class_ratio,
            num_objects = len(set(object_ids)),
            num_objects_per_class = num_objects_per_class,
            decision_class = dominant_class_value
        )

        if (self.max_depth is None or depth < self.max_depth) and dominant_class_ratio < self.pruning_factor:
            best_cut = self._get_best_temp_cut(X, y, object_ids, time_points, attributes_names, alpha_param_start,
                                                                alpha_step, alpha_step_count, quality_measure,
                                                                quality_sel_method)
            if best_cut is None:
                return node

            node.attribute_name = best_cut['attributeName']
            node.attribute_index = best_cut['attributeIndex']
            node.attribute_value = best_cut['attributeValue']
            node.quality = best_cut['quality']
            node.is_up_cut = best_cut['isUpCut']
            node.alpha_of_time_points = best_cut['alphaOfTimePoints']
            node.ids_obj_over_cut = best_cut['idsObjOverCut']

            ids_over_cut = list(best_cut['idsObjOverCut'])
            tp_count = dict(Counter(object_ids))
            object_ids_list = list(object_ids)
            indexes_over_cut = []
            for ido in ids_over_cut:
                for i in range(object_ids_list.index(ido), object_ids_list.index(ido) + tp_count[ido]):
                    indexes_over_cut.append(i)
            ids_under_cut = [item for item in set(object_ids_list) if item not in ids_over_cut]
            indexes_under_cut = []
            for idu in ids_under_cut:
                for i in range(object_ids_list.index(idu), object_ids_list.index(idu) + tp_count[idu]):
                    indexes_under_cut.append(i)

            left_X = X[indexes_over_cut]
            left_y = y[indexes_over_cut]
            left_object_ids = object_ids[indexes_over_cut]
            left_time_points = time_points[indexes_over_cut]
            right_X = X[indexes_under_cut]
            right_y = y[indexes_under_cut]
            right_object_ids = object_ids[indexes_under_cut]
            right_time_points = time_points[indexes_under_cut]

            node.left = self._grow_tree(left_X, left_y, left_object_ids, left_time_points, attributes_names,
                                        alpha_param_start, alpha_step, alpha_step_count, quality_measure,
                                        quality_sel_method, depth + 1)
            node.right = self._grow_tree(right_X, right_y, right_object_ids, right_time_points, attributes_names,
                                         alpha_param_start, alpha_step, alpha_step_count, quality_measure,
                                         quality_sel_method, depth + 1)
        return node


    def _get_best_temp_cut(self, X, y, object_ids, time_points, attributes_names, alpha_param_start, alpha_step,
                           alpha_step_count, quality_measure, quality_sel_method):

        best_cuts_list = []
        time_points_count = dict(Counter(object_ids))
        dec_values = list(set(y))
        obj_over_under_cut_counter = {}
        num_objects_per_class = self._num_obj_per_class(object_ids, y)
        for dv in dec_values:
            obj_over_under_cut_counter[str(dv) + "_UNDER"] = num_objects_per_class[dv]
            obj_over_under_cut_counter[str(dv) + "_OVER"] = 0

        for attr_index in range(self.n_attributes_):
            data2 = np.array([X[:, attr_index], object_ids, y]).transpose()
            records = data2[np.lexsort((data2[:, 1], data2[:, 0]))]  # sort

            if alpha_step <= 0:
                alpha_step = 1.0 / max(time_points_count.values()) # 1/max_tp
                alpha_step = math.floor(alpha_step * 10 ** 16 + 0.5) / 10 ** 16
            if alpha_step_count < 0:
                alpha_step_count = 0
            if alpha_step_count >= math.floor((1.0 - alpha_param_start) / alpha_step):
                alpha_step_count = int(math.floor((1.0 - alpha_param_start) / alpha_step))

            ###############
            for a in range(0, alpha_step_count + 1):
                alpha_param = alpha_param_start + a * alpha_step
                scale = max(len(str(alpha_step)) - 2, len(str(alpha_param_start)) - 2)
                alpha_param = math.floor(alpha_param * 10 ** scale + 0.5) / 10 ** scale
                min_time_points = {k: v * alpha_param for k, v in time_points_count.items()}

                # --
                obj_over_under_down_cut_count = obj_over_under_cut_counter.copy()
                time_points_over_down_cut = dict()
                v_attr = records[0, 0]
                id_o = records[0, 1]
                v_dec = records[0, 2]
                ids_moved_over = set()

                for i in range(1, len(records)):
                    if (str(id_o) + str(v_dec)) not in time_points_over_down_cut.keys():
                        time_points_over_down_cut[(str(id_o) + str(v_dec))] = 1
                    else:
                        time_points_over_down_cut[(str(id_o) + str(v_dec))] += 1

                    id_o_act = records[i, 1]
                    v_dec_act = records[i, 2]

                    if id_o != id_o_act:
                        tp_count = time_points_over_down_cut[str(id_o) + str(v_dec)]
                        if tp_count >= min_time_points[id_o] and id_o not in ids_moved_over:
                            obj_over_under_down_cut_count[str(v_dec) + "_OVER"] += 1
                            obj_over_under_down_cut_count[str(v_dec) + "_UNDER"] -= 1
                            ids_moved_over.add(id_o)
                        id_o = id_o_act
                        v_dec = v_dec_act

                    v_attr_act = records[i, 0]
                    if v_attr != v_attr_act:
                        cut_value = (v_attr + v_attr_act) / 2

                        discerned_pairs_from_different_classes = 0
                        discerned_pairs_from_the_same_class = 0

                        cut_quality = 0
                        for j in range(0, len(dec_values)):
                            sum_over = 0
                            sum_under = 0
                            for k in range(0, len(dec_values)):
                                if j == k:
                                    continue
                                sum_over += obj_over_under_down_cut_count[str(dec_values[k]) + "_OVER"]
                                sum_under += obj_over_under_down_cut_count[str(dec_values[k]) + "_UNDER"]
                            discerned_pairs_from_different_classes += obj_over_under_down_cut_count[
                                                                      str(dec_values[j]) + "_OVER"] * sum_under

                            discerned_pairs_from_the_same_class += obj_over_under_down_cut_count[str(dec_values[j]) + "_OVER"] * \
                                                              obj_over_under_down_cut_count[str(dec_values[j]) + "_UNDER"]

                        if quality_measure == TemporalCutsQualityMeasure.DISC_PAIR_FROM_DIFF_CLASSES:
                            cut_quality = discerned_pairs_from_different_classes
                        else:
                            if quality_measure == TemporalCutsQualityMeasure.DISC_PAIR_FROM_THE_SAME_CLASS:
                                cut_quality = discerned_pairs_from_the_same_class
                            else:
                                if quality_measure == TemporalCutsQualityMeasure.D_PAIR_DIFF_CLASSES_BY_D_PAIR_THE_SAME_CLASS:
                                    if discerned_pairs_from_the_same_class > 0:
                                        cut_quality = discerned_pairs_from_different_classes / discerned_pairs_from_the_same_class
                                    else:
                                        cut_quality = discerned_pairs_from_different_classes + 1

                        if cut_quality > 0:
                            temp_cut_down = {'attributeName' : attributes_names[attr_index], 'attributeIndex' : attr_index, 'attributeValue' : cut_value, 'quality' : cut_quality,
                                           'isUpCut' : False, 'alphaOfTimePoints' : alpha_param, 'idsObjOverCut' : ids_moved_over.copy()}

                            if len(best_cuts_list) == 0:
                                best_cuts_list.append(temp_cut_down)
                            else:
                                best_cuts_list.append(temp_cut_down)
                                sorted_all_cuts = []
                                if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY:
                                    sorted_all_cuts = sorted(best_cuts_list, key=lambda x: x['quality'],
                                                             reverse=True)
                                else:
                                    if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY_MAX_ALPHA:
                                        sorted_all_cuts = sorted(best_cuts_list,
                                                                 key=lambda x: (x['quality'], x['alphaOfTimePoints']),
                                                                 reverse=True)
                                    else:
                                        if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY_MIN_ALPHA:
                                            sorted_all_cuts = sorted(best_cuts_list, key=lambda x: (-x['quality'], x[
                                                'alphaOfTimePoints']))
                                best_cuts_list = [sorted_all_cuts[0]]

                        v_attr = v_attr_act

                #--
                obj_over_under_up_cut_count = obj_over_under_cut_counter.copy()
                time_points_over_up_cut = dict()
                v_attr = records[len(records) - 1, 0]
                id_o = records[len(records) - 1, 1]
                v_dec = records[len(records) - 1, 2]
                ids_moved_over = set()

                for i in range(len(records) - 2, 0 - 1, -1):
                    if (str(id_o) + str(v_dec)) not in time_points_over_up_cut.keys():
                        time_points_over_up_cut[(str(id_o) + str(v_dec))] = 1
                    else:
                        time_points_over_up_cut[(str(id_o) + str(v_dec))] += 1

                    id_o_act = records[i, 1]
                    v_dec_act = records[i, 2]

                    if id_o != id_o_act:
                        tp_count = time_points_over_up_cut[str(id_o) + str(v_dec)]
                        if tp_count >= min_time_points[id_o] and id_o not in ids_moved_over:
                            obj_over_under_up_cut_count[str(v_dec) + "_OVER"] += 1
                            obj_over_under_up_cut_count[str(v_dec) + "_UNDER"] -= 1
                            ids_moved_over.add(id_o)
                        id_o = id_o_act
                        v_dec = v_dec_act

                    v_attr_act = records[i, 0]
                    if v_attr != v_attr_act:
                        cut_value = (v_attr + v_attr_act) / 2

                        discerned_pairs_from_different_classes = 0
                        discerned_pairs_from_the_same_class = 0

                        cut_quality = 0
                        for j in range(0, len(dec_values)):
                            sum_over = 0
                            sum_under = 0
                            for k in range(0, len(dec_values)):
                                if j == k:
                                    continue
                                sum_over += obj_over_under_up_cut_count[str(dec_values[k]) + "_OVER"]
                                sum_under += obj_over_under_up_cut_count[str(dec_values[k]) + "_UNDER"]
                            discerned_pairs_from_different_classes += obj_over_under_up_cut_count[
                                                                      str(dec_values[j]) + "_OVER"] * sum_under

                            discerned_pairs_from_the_same_class += obj_over_under_up_cut_count[str(dec_values[j]) + "_OVER"] * \
                                                              obj_over_under_up_cut_count[str(dec_values[j]) + "_UNDER"]

                        if quality_measure == TemporalCutsQualityMeasure.DISC_PAIR_FROM_DIFF_CLASSES:
                            cut_quality = discerned_pairs_from_different_classes
                        else:
                            if quality_measure == TemporalCutsQualityMeasure.DISC_PAIR_FROM_THE_SAME_CLASS:
                                cut_quality = discerned_pairs_from_the_same_class
                            else:
                                if quality_measure == TemporalCutsQualityMeasure.D_PAIR_DIFF_CLASSES_BY_D_PAIR_THE_SAME_CLASS:
                                    if discerned_pairs_from_the_same_class > 0:
                                        cut_quality = discerned_pairs_from_different_classes / discerned_pairs_from_the_same_class
                                    else:
                                        cut_quality = discerned_pairs_from_different_classes + 1

                        if cut_quality > 0:
                            temp_cut_up = {'attributeName' : attributes_names[attr_index], 'attributeIndex': attr_index, 'attributeValue': cut_value, 'quality': cut_quality,
                                           'isUpCut': True, 'alphaOfTimePoints': alpha_param,
                                           'idsObjOverCut': ids_moved_over.copy()}

                            if len(best_cuts_list) == 0:
                                best_cuts_list.append(temp_cut_up)
                            else:
                                best_cuts_list.append(temp_cut_up)
                                sorted_all_cuts = []
                                if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY:
                                    sorted_all_cuts = sorted(best_cuts_list, key=lambda x: x['quality'],
                                                             reverse=True)
                                else:
                                    if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY_MAX_ALPHA:
                                        sorted_all_cuts = sorted(best_cuts_list,
                                                                 key=lambda x: (x['quality'], x['alphaOfTimePoints']),
                                                                 reverse=True)
                                    else:
                                        if quality_sel_method == BestTemporalCutSelectionMethod.MAX_QUALITY_MIN_ALPHA:
                                            sorted_all_cuts = sorted(best_cuts_list, key=lambda x: (-x['quality'], x[
                                                'alphaOfTimePoints']))
                                best_cuts_list = [sorted_all_cuts[0]]

                        v_attr = v_attr_act

        if len(best_cuts_list) == 0:
            return None
        return best_cuts_list[0]


    def print_tree(self):
        self._print_tree_recursively(self.tree_, depth=0)


    def _print_tree_recursively(self, node, depth):
        if node is not None:
            if node.is_up_cut:
                sign = '>='
            else:
                sign = '<'
            txt = "   " * depth + "|--"
            if node.attribute_name is not None:
                txt += str(node.attribute_name) + " " + sign + " " + str(node.attribute_value) \
                  + " @ >= " + str(node.alpha_of_time_points)
            txt += " class = " + str(node.decision_class) + "(" + str(round(node.dominant_class_ratio*100, 3)) + "%)"
            txt += "   {nO:" + str(node.num_objects) + " " + str(node.num_objects_per_class) + "}"
            print(txt)
            self._print_tree_recursively(node.left, depth + 1)
            self._print_tree_recursively(node.right, depth + 1)

