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

class CutsConflictsResolvingMethod(Enum):
    CRM_P = 1
    CRM_T = 2

class VerifyingTemporalCut:     ###################################################################################
    def __init__(self):
        self.attribute_name = None
        self.attribute_index = None
        self.attribute_value = None
        self.is_up_cut = None # czy a >= v
        self.alpha_of_time_points = None
        self.quality = None
        self.ids_obj_over_cut = None
        self.ids_obj_under_cut = None
    def __str__(self):
        if self.is_up_cut:
            sign = '>='
        else:
            sign = '<'
        str_v_cut = 'V_CUT:' + self.attribute_name + sign + str(self.attribute_value) + ' alpha=' \
                    + str(self.alpha_of_time_points) + ' quality=' + str(self.quality) + ' ids_over:' \
                    + str(self.ids_obj_over_cut) + ' ids_under:' + str(self.ids_obj_under_cut)
        return str_v_cut
    def print_cut(self):
        print('v_cut:\n\tattribute_name', self.attribute_name, '\n\tattribute_index', self.attribute_index,
              '\n\tattribute_value', self.attribute_value, '\n\tis_up_cut', self.is_up_cut,
              '\n\talpha_of_time_points', self.alpha_of_time_points, '\n\tquality', self.quality,
              '\n\tids_obj_over_cut', self.ids_obj_over_cut, '\n\tids_obj_under_cut', self.ids_obj_under_cut)
    def print_cut_short(self):
        if self.is_up_cut:
            sign = '>='
        else:
            sign = '<'
        print('v_cut:', self.attribute_name, sign, self.attribute_value,
              'alpha=', self.alpha_of_time_points, 'quality=', self.quality,
              'ids_over:', self.ids_obj_over_cut, 'ids_under:', self.ids_obj_under_cut)

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
        self.ids_obj_under_cut = None
        self.verifying_cuts = None
        self.left = None
        self.right = None

class VerifyingTemporalCutsDecisionTreeClassifier:
    def __init__(self, max_depth=None, pruning_factor=1.0):
        self.vcuts_max_count = None
        self.min_vc_quality_ratio = None
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
        self.class_to_index_ = None
        self.n_attributes_ = None


    def fit(self, data_tr, alpha_start=1.0, alpha_step=0.0, alpha_step_count=0,
            quality_measure=TemporalCutsQualityMeasure.DISC_PAIR_FROM_DIFF_CLASSES,
            quality_sel_method = BestTemporalCutSelectionMethod.MAX_QUALITY,
            min_vc_quality_ratio = 1.0, vcuts_max_count = 0):
        no_column = data_tr.shape[1]  # Ustalenie liczby kolumn w danych
        features_tr = np.array(data_tr.iloc[:, 2:no_column - 1])  # Wyodrębnienie częśći warunkowej danych
        labels_tr = np.array(data_tr.iloc[:, no_column - 1])  # Wyodrębnienie kolumny decyzyjnej
        self.classes_ = np.unique(labels_tr)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
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
        self.min_vc_quality_ratio = min_vc_quality_ratio
        self.vcuts_max_count = vcuts_max_count
        self.tree_ = self._grow_tree(features_tr, labels_tr, self.object_ids, self.time_points, self.attributes_names,
                                     self.alpha_param_start, self.alpha_step, self.alpha_step_count,
                                     self.quality_measure, self.quality_sel_method,
                                     self.min_vc_quality_ratio, self.vcuts_max_count)


    def predict_proba(self, data_tst, conflicts_resolv_method):
        id_col = data_tst.columns[0]
        data_tst[id_col] = data_tst[id_col].astype(str)
        probas = []
        for tw_id, df_tw in data_tst.groupby(id_col, sort=False):
            proba = self._predict_proba_single_tw(df_tw, conflicts_resolv_method)
            probas.append(proba)
        return np.array(probas)


    def _predict_proba_single_tw(self, window_tst, conflicts_resolv_method):
        no_column = window_tst.shape[1]
        inputs = np.array(window_tst.iloc[:, 2:no_column - 1])
        tp_count = inputs.shape[0]
        node = self.tree_
        dec, conf = self._predict_using_verifying_cuts(node, inputs, tp_count, 0, conflicts_resolv_method)
        conf = self._normalize_confidence(conf)
        n_classes = len(self.classes_)
        probs = np.zeros(n_classes)
        class_idx = self.class_to_index_[dec]

        if n_classes == 1:
            probs[0] = 1.0
            return probs

        probs[class_idx] = conf
        remaining = 1.0 - conf
        other_classes = n_classes - 1

        for i in range(n_classes):
            if i != class_idx:
                probs[i] = remaining / other_classes

        return probs


    def _normalize_confidence(self, conf):
        if conf is None or np.isnan(conf):
            return 1.0 / len(self.classes_)
        if 0.0 <= conf <= 1.0:
            return conf
        return conf / (1.0 + conf)


    def predict(self, data_tst, conflicts_resolv_method):
        id_col = data_tst.columns[0]
        predicted = []
        for tw_id, df_tw in data_tst.groupby(id_col, sort=False):
            predicted.append(self._predict_single_tw_using_verifying_cuts(df_tw, conflicts_resolv_method))
        return predicted


    def _predict_single_tw_using_verifying_cuts(self, window_tst, conflicts_resolv_method):
        no_column = window_tst.shape[1]
        inputs = np.array(window_tst.iloc[:, 2:no_column - 1])
        tp_count = inputs.shape[0]
        node = self.tree_
        dec, conf = self._predict_using_verifying_cuts(node, inputs, tp_count, 0, conflicts_resolv_method)
        return dec


    def _predict_using_verifying_cuts(self, node, inputs, tp_count, depth, conflicts_resolv_method):
        if node.left is None:
            return node.decision_class, 1.0

        n_patterns = len(node.verifying_cuts) + 1
        tp_over = 0
        tp_under = 0

        n_tp_matched_patterns = []
        for i in range(n_patterns):
            n_tp_matched_patterns.append(0)

        for tp in inputs:
            tp_patt_match = 0
            if node.is_up_cut:
                if tp[node.attribute_index] >= node.attribute_value:
                    tp_patt_match += 1
                    n_tp_matched_patterns[0] += 1
            else:
                if tp[node.attribute_index] < node.attribute_value:
                    tp_patt_match += 1
                    n_tp_matched_patterns[0] += 1

            for vc in node.verifying_cuts:
                vc_n = 1
                if vc.is_up_cut:
                    if tp[vc.attribute_index] >= vc.attribute_value:
                        tp_patt_match += 1
                        n_tp_matched_patterns[vc_n] += 1
                else:
                    if tp[vc.attribute_index] < vc.attribute_value:
                        tp_patt_match += 1
                        n_tp_matched_patterns[vc_n] += 1
                vc_n += 1

            if tp_patt_match == n_patterns:
                tp_over += 1
            elif tp_patt_match == 0:
                tp_under += 1

        if tp_over / tp_count >= node.alpha_of_time_points:
            return self._predict_using_verifying_cuts(node.left, inputs, tp_count, depth + 1, conflicts_resolv_method)
        elif tp_under / tp_count > 1.0 - node.alpha_of_time_points:
            return self._predict_using_verifying_cuts(node.right, inputs, tp_count, depth + 1, conflicts_resolv_method)
        else:
            d1, c1 = self._predict_using_verifying_cuts(node.left, inputs, tp_count, depth + 1, conflicts_resolv_method)
            d2, c2 = self._predict_using_verifying_cuts(node.right, inputs, tp_count, depth + 1, conflicts_resolv_method)
            if d1 == d2:
                return d1, max(c1, c2)
            else:
                p1 = 0
                p2 = 0

                eps_a = 1e-12
                alpha_tp = min(node.alpha_of_time_points, 1.0 - eps_a)
                if conflicts_resolv_method == CutsConflictsResolvingMethod.CRM_P:
                    p1 = tp_over / tp_count / alpha_tp
                    p2 = tp_under / tp_count / (1 - alpha_tp)

                if conflicts_resolv_method == CutsConflictsResolvingMethod.CRM_T:
                    n_patt_matched = 0
                    n_patt_not_matched = 0
                    for tpm in n_tp_matched_patterns:
                        if tpm / tp_count >= node.alpha_of_time_points:
                            n_patt_matched += 1
                        elif (tp_count - tpm) / tp_count > 1.0 - node.alpha_of_time_points:
                            n_patt_not_matched += 1
                    p1 = n_patt_matched / n_patterns
                    p2 = n_patt_not_matched / n_patterns

                if p1 > p2:
                    return d1, p1
                elif p1 < p2:
                    return d2, p2
                else:  # p1 == p2
                    return self._predict_by_main_cut(inputs, tp_count), p1


    def _predict_by_main_cut(self, inputs, tp_count):
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
        return maximum / count


    def _dominant_class_value(self, ids, y):
        num_obj_per_class = self._num_obj_per_class(ids, y)
        max_key = max(num_obj_per_class, key=num_obj_per_class.get)
        return float(max_key)


    def _grow_tree(self, X, y, object_ids, time_points, attributes_names, alpha_param_start, alpha_step,
                   alpha_step_count, quality_measure, quality_sel_method, min_vc_quality_ratio, vcuts_max_count, depth=0):
        num_objects_per_class = list(self._num_obj_per_class(object_ids, y).values())
        dominant_class_ratio = self._dominant_class_ratio(object_ids, y)
        dominant_class_value = self._dominant_class_value(object_ids, y)

        node = TemporalCutsTreeNode(
            dominant_class_ratio=dominant_class_ratio,
            num_objects=len(set(object_ids)),
            num_objects_per_class=num_objects_per_class,
            decision_class=dominant_class_value
        )

        if (self.max_depth is None or depth < self.max_depth) and dominant_class_ratio < self.pruning_factor:
            # wybor najlepszego ciecia
            best_cut = self._get_best_temp_cut(X, y, object_ids, time_points, attributes_names, alpha_param_start,
                                               alpha_step, alpha_step_count, quality_measure, quality_sel_method)
            if best_cut is None:
                return node

            node.attribute_name = best_cut['attributeName']
            node.attribute_index = best_cut['attributeIndex']
            node.attribute_value = best_cut['attributeValue']
            node.quality = best_cut['quality']
            node.is_up_cut = best_cut['isUpCut']
            node.alpha_of_time_points = best_cut['alphaOfTimePoints']
            node.ids_obj_over_cut = best_cut['idsObjOverCut']

            v_cuts_list = self._get_verifying_temp_cuts(best_cut, X, y, object_ids, attributes_names,
                                                        quality_measure, min_vc_quality_ratio, vcuts_max_count)
            node.verifying_cuts = v_cuts_list

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
                                        quality_sel_method, min_vc_quality_ratio, vcuts_max_count, depth + 1)
            node.right = self._grow_tree(right_X, right_y, right_object_ids, right_time_points, attributes_names,
                                         alpha_param_start, alpha_step, alpha_step_count, quality_measure,
                                         quality_sel_method, min_vc_quality_ratio, vcuts_max_count, depth + 1)
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
            records = data2[np.lexsort((data2[:, 1], data2[:, 0]))]
            if alpha_step <= 0:
                alpha_step = 1.0 / max(time_points_count.values())  # 1/max_tp
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

                            discerned_pairs_from_the_same_class += (obj_over_under_down_cut_count[str(dec_values[j]) + "_OVER"] *
                                                                    obj_over_under_down_cut_count[str(dec_values[j]) + "_UNDER"])

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
                            temp_cut_down = {'attributeName': attributes_names[attr_index], 'attributeIndex': attr_index,
                                           'attributeValue': cut_value, 'quality': cut_quality,
                                           'isUpCut': False, 'alphaOfTimePoints': alpha_param,
                                           'idsObjOverCut': ids_moved_over.copy()}

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
                        if tp_count >= min_time_points[
                            id_o] and id_o not in ids_moved_over:
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

                            discerned_pairs_from_the_same_class += (obj_over_under_up_cut_count[str(dec_values[j]) + "_OVER"] *
                                                                    obj_over_under_up_cut_count[str(dec_values[j]) + "_UNDER"])

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
                            temp_cut_up = {'attributeName': attributes_names[attr_index], 'attributeIndex': attr_index,
                                         'attributeValue': cut_value, 'quality': cut_quality,
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


    def __get_best_v_cuts_on_attr(self, v_cuts_on_attr, dec_values, alphaParam, quality_measure, min_vc_quality, get_only_one):
        best_v_cuts_list = []
        for c in v_cuts_on_attr:
            discerned_pairs_from_different_classes = 0
            discerned_pairs_from_the_same_class = 0
            cut_quality = 0
            tw_over = v_cuts_on_attr[c][0]
            tw_under = v_cuts_on_attr[c][1]
            for j in range(0, len(dec_values)):
                sum_over = 0
                sum_under = 0
                for k in range(0, len(dec_values)):
                    if j == k:
                        continue
                    sum_over += tw_over[str(dec_values[k])]
                    sum_under += tw_under[str(dec_values[k])]
                discerned_pairs_from_different_classes += tw_over[str(dec_values[j])] * sum_under
                discerned_pairs_from_the_same_class += tw_over[str(dec_values[j])] * tw_under[str(dec_values[j])]

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
            v_c = c
            v_c_stats = v_cuts_on_attr[c]
            v_c_quality = cut_quality

            if cut_quality >= min_vc_quality:
                v_cut = VerifyingTemporalCut()
                v_cut.attribute_name = v_c[0]
                v_cut.attribute_index = v_c[1]
                v_cut.attribute_value = v_c[2]
                if v_c[3] == 'UP':
                    v_cut.is_up_cut = True
                else:
                    v_cut.is_up_cut = False
                v_cut.alpha_of_time_points = alphaParam
                v_cut.quality = v_c_quality
                v_cut.ids_obj_over_cut = v_c_stats[2]
                v_cut.ids_obj_under_cut = v_c_stats[3]

                best_v_cuts_list.append(v_cut)

        best_v_cuts_list.sort(key=lambda cut: cut.quality, reverse=True)
        if get_only_one:
            return [best_v_cuts_list[0]]
        return best_v_cuts_list


    def _get_verifying_temp_cuts(self, primary_cut, X, y, object_ids, attributes_names, quality_measure,
                                 min_vc_quality_ratio, v_cuts_max_count):
        best_v_cuts_list = []
        time_points_count = dict(Counter(object_ids))
        dec_values = list(set(y))
        alpha_param = primary_cut['alphaOfTimePoints']
        min_time_points = {k: v * alpha_param for k, v in time_points_count.items()}

        tw_over_pC_vC_DOWN_counter = dict()
        tw_under_pC_vC_DOWN_counter = dict()
        tw_over_pC_vC_UP_counter = dict()
        tw_under_pC_vC_UP_counter = dict()
        for dv in dec_values:
            tw_over_pC_vC_DOWN_counter[str(dv)] = 0
            tw_under_pC_vC_DOWN_counter[str(dv)] = 0
            tw_over_pC_vC_UP_counter[str(dv)] = 0
            tw_under_pC_vC_UP_counter[str(dv)] = 0

        for attr_index in range(self.n_attributes_):
            v_cuts_on_attr = dict()
            if attr_index == primary_cut['attributeIndex']:
                continue

            data2 = np.array([X[:, attr_index], object_ids, X[:, primary_cut['attributeIndex']], y]).transpose()
            records = data2[np.lexsort((data2[:, 1], data2[:, 0]))]

        ###############
            tw_over_pC_vC_DOWN_count = tw_over_pC_vC_DOWN_counter.copy()
            tw_under_pC_vC_UP_count = tw_under_pC_vC_UP_counter.copy()
            tp_over_pC_vC_DOWN = dict()
            tp_under_pC_vC_UP = dict()
            v_attr = records[0, 0]
            id_o = records[0, 1]
            v_dec = records[0, 3]
            ids_moved_over_pC_vC_DOWN = set()
            ids_moved_under_pC_vC_UP = set()

            for i in range(1, len(records)):
                if v_attr == records[-1, 0]:
                    continue
                if primary_cut["isUpCut"]:
                    if records[i - 1, 2] >= primary_cut["attributeValue"]:
                        if (str(id_o)) not in tp_over_pC_vC_DOWN.keys():
                            tp_over_pC_vC_DOWN[str(id_o)] = 1
                        else:
                            tp_over_pC_vC_DOWN[str(id_o)] += 1

                        if (str(id_o)) not in tp_under_pC_vC_UP.keys():
                            tp_under_pC_vC_UP[str(id_o)] = 0
                    else:
                        if (str(id_o)) not in tp_over_pC_vC_DOWN.keys():
                            tp_over_pC_vC_DOWN[str(id_o)] = 0
                        if (str(id_o)) not in tp_under_pC_vC_UP.keys():
                            tp_under_pC_vC_UP[str(id_o)] = 1
                        else:
                            tp_under_pC_vC_UP[str(id_o)] += 1
                else:
                    if records[i - 1, 2] < primary_cut["attributeValue"]:
                        if (str(id_o)) not in tp_over_pC_vC_DOWN.keys():
                            tp_over_pC_vC_DOWN[str(id_o)] = 1
                        else:
                            tp_over_pC_vC_DOWN[str(id_o)] += 1

                        if (str(id_o)) not in tp_under_pC_vC_UP.keys():
                            tp_under_pC_vC_UP[str(id_o)] = 0
                    else:
                        if (str(id_o)) not in tp_over_pC_vC_DOWN.keys():
                            tp_over_pC_vC_DOWN[str(id_o)] = 0
                        if (str(id_o)) not in tp_under_pC_vC_UP.keys():
                            tp_under_pC_vC_UP[str(id_o)] = 1
                        else:
                            tp_under_pC_vC_UP[str(id_o)] += 1

                id_o_act = records[i, 1]
                v_dec_act = records[i, 3]
                if id_o != id_o_act:
                    tp_count_DOWN = tp_over_pC_vC_DOWN[str(id_o)]
                    tp_count_UP = tp_under_pC_vC_UP[str(id_o)]
                    if tp_count_DOWN >= min_time_points[id_o] and id_o not in ids_moved_over_pC_vC_DOWN:
                        tw_over_pC_vC_DOWN_count[str(v_dec)] += 1
                        ids_moved_over_pC_vC_DOWN.add(id_o)
                    if (tp_count_UP > time_points_count[id_o] * (1-primary_cut["alphaOfTimePoints"])
                            and id_o not in ids_moved_under_pC_vC_UP):
                        tw_under_pC_vC_UP_count[str(v_dec)] += 1
                        ids_moved_under_pC_vC_UP.add(id_o)
                    id_o = id_o_act
                    v_dec = v_dec_act

                v_attr_act = records[i, 0]
                if v_attr != v_attr_act:
                    cut_value = (v_attr + v_attr_act) / 2
                    key_vC_DOWN = (attributes_names[attr_index], attr_index, cut_value, 'DOWN')
                    value_vC_DOWN = [tw_over_pC_vC_DOWN_count.copy(), {}, ids_moved_over_pC_vC_DOWN.copy(), set()]
                    v_cuts_on_attr[key_vC_DOWN] = value_vC_DOWN
                    key_vC_UP = (attributes_names[attr_index], attr_index, cut_value, 'UP')
                    value_vC_UP = [{}, tw_under_pC_vC_UP_count.copy(), set(), ids_moved_under_pC_vC_UP.copy()]
                    v_cuts_on_attr[key_vC_UP] = value_vC_UP
                    v_attr = v_attr_act


        ###############
            tw_under_pC_vC_DOWN_count = tw_under_pC_vC_DOWN_counter.copy()
            tw_over_pC_vC_UP_count = tw_over_pC_vC_UP_counter.copy()
            tp_under_pC_vC_DOWN = dict()
            tp_over_pC_vC_UP = dict()
            v_attr = records[len(records) - 1, 0]
            id_o = records[len(records) - 1, 1]
            v_dec = records[len(records) - 1, 3]
            ids_moved_under_pC_vC_DOWN = set()
            ids_moved_over_pC_vC_UP = set()

            for i in range(len(records) - 2, 0 - 1, -1):
                if v_attr == records[0, 0]:
                    continue
                if primary_cut["isUpCut"]:
                    if records[i + 1, 2] >= primary_cut["attributeValue"]:
                        if (str(id_o)) not in tp_over_pC_vC_UP.keys():
                            tp_over_pC_vC_UP[str(id_o)] = 1
                        else:
                            tp_over_pC_vC_UP[str(id_o)] += 1

                        if (str(id_o)) not in tp_under_pC_vC_DOWN.keys():
                            tp_under_pC_vC_DOWN[str(id_o)] = 0
                    else:
                        if (str(id_o)) not in tp_over_pC_vC_UP.keys():
                            tp_over_pC_vC_UP[str(id_o)] = 0
                        if (str(id_o)) not in tp_under_pC_vC_DOWN.keys():
                            tp_under_pC_vC_DOWN[str(id_o)] = 1
                        else:
                            tp_under_pC_vC_DOWN[str(id_o)] += 1
                else:
                    if records[i + 1, 2] < primary_cut["attributeValue"]:
                        if (str(id_o)) not in tp_over_pC_vC_UP.keys():
                            tp_over_pC_vC_UP[str(id_o)] = 1
                        else:
                            tp_over_pC_vC_UP[str(id_o)] += 1

                        if (str(id_o)) not in tp_under_pC_vC_DOWN.keys():
                            tp_under_pC_vC_DOWN[str(id_o)] = 0
                    else:
                        if (str(id_o)) not in tp_over_pC_vC_UP.keys():
                            tp_over_pC_vC_UP[str(id_o)] = 0
                        if (str(id_o)) not in tp_under_pC_vC_DOWN.keys():
                            tp_under_pC_vC_DOWN[str(id_o)] = 1
                        else:
                            tp_under_pC_vC_DOWN[str(id_o)] += 1

                id_o_act = records[i, 1]
                v_dec_act = records[i, 3]
                if id_o != id_o_act:
                    tp_count_DOWN = tp_under_pC_vC_DOWN[str(id_o)]
                    tp_count_UP = tp_over_pC_vC_UP[str(id_o)]
                    if tp_count_DOWN > time_points_count[id_o] * (1 - primary_cut["alphaOfTimePoints"]) \
                            and id_o not in ids_moved_under_pC_vC_DOWN:
                        tw_under_pC_vC_DOWN_count[str(v_dec)] += 1
                        ids_moved_under_pC_vC_DOWN.add(id_o)
                    if tp_count_UP >= min_time_points[id_o] and id_o not in ids_moved_over_pC_vC_UP:
                        tw_over_pC_vC_UP_count[str(v_dec)] += 1
                        ids_moved_over_pC_vC_UP.add(id_o)
                    id_o = id_o_act
                    v_dec = v_dec_act

                v_attr_act = records[i, 0]
                if v_attr != v_attr_act:
                    cut_value = (v_attr + v_attr_act) / 2
                    key_vC_DOWN = (attributes_names[attr_index], attr_index, cut_value, 'DOWN')
                    v_cuts_on_attr[key_vC_DOWN][1] = tw_under_pC_vC_DOWN_count.copy()
                    v_cuts_on_attr[key_vC_DOWN][3] = ids_moved_under_pC_vC_DOWN.copy()
                    key_vC_UP = (attributes_names[attr_index], attr_index, cut_value, 'UP')
                    v_cuts_on_attr[key_vC_UP][0] = tw_over_pC_vC_UP_count.copy()
                    v_cuts_on_attr[key_vC_UP][2] = ids_moved_over_pC_vC_UP.copy()
                    v_attr = v_attr_act

            min_vc_quality = min_vc_quality_ratio * primary_cut['quality']
            best_attr_vCuts_list = self.__get_best_v_cuts_on_attr(v_cuts_on_attr, dec_values, alpha_param, quality_measure, min_vc_quality, False)

            if best_attr_vCuts_list:
                best_v_cuts_list.append(best_attr_vCuts_list[0])

        best_v_cuts_list.sort(key=lambda cut: cut.quality, reverse=True)

        if v_cuts_max_count < 1:
            v_cuts_max_count = 1
        return best_v_cuts_list[:v_cuts_max_count]


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
            txt += "   {nTW:" + str(node.num_objects) + " perClass:" + str(node.num_objects_per_class) + "}\n"
            v_cuts = node.verifying_cuts
            if v_cuts is not None:
                str_tab = "   " * depth + "  "
                str_v_cut = ''
                for vc in v_cuts:
                    if vc.is_up_cut:
                        sign = '>='
                    else:
                        sign = '<'
                    str_v_cut += str_tab + 'v_c: ' + vc.attribute_name + sign + str(vc.attribute_value) + ' @>=' \
                                 + str(vc.alpha_of_time_points) + ' q=' + str(vc.quality) + '\n'
                txt += str_v_cut
            print(txt[:-1])
            self._print_tree_recursively(node.left, depth + 1)
            self._print_tree_recursively(node.right, depth + 1)


