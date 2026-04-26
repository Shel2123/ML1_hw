import numpy as np
from collections import Counter


def find_best_split(
        feature_vector: np.ndarray | list, target_vector: np.ndarray| list
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    x = np.array(feature_vector)
    y = np.array(target_vector)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("target_vector consists of more than 2 classes")

    thresholds = np.array([])
    ginis = np.array([])
    threshold_best = None
    gini_best = None

    order = x.argsort()
    x_sorted = x[order]
    y_sorted = y[order]

    n = len(x_sorted)

    diff = x_sorted[1:] != x_sorted[:-1]
    valid_idx = np.where(diff)[0]

    if len(valid_idx) <= 0:
        return thresholds, ginis, threshold_best, gini_best

    thresholds = (x_sorted[valid_idx] + x_sorted[valid_idx + 1]) / 2
    y_cumsum = np.cumsum(y_sorted)

    total_ones = y_cumsum[-1]
    left_ones = y_cumsum[valid_idx]
    right_ones = total_ones - left_ones

    left_size = valid_idx + 1
    right_size = n - left_size

    p1_left = left_ones / left_size
    p0_left = 1 - p1_left

    p1_right = right_ones / right_size
    p0_right = 1 - p1_right

    gini_left = 1 - p1_left ** 2 - p0_left ** 2
    gini_right = 1 - p1_right ** 2 - p0_right ** 2
    ginis = - (left_size / n) * gini_left - (right_size / n) * gini_right

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self._int_to_class = None
        self._class_to_int = None
        self._classes = None
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            self._make_terminal(sub_y, node)
            return

        if self._max_depth is not None and depth >= self._max_depth:
            self._make_terminal(sub_y, node)
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            self._make_terminal(sub_y, node)
            return


        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature]) 
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            current_split = feature_vector < threshold

            if self._min_samples_leaf is not None:
                if np.sum(current_split) < self._min_samples_leaf:
                    continue
                if np.sum(~current_split) < self._min_samples_leaf:
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = current_split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        y = np.array(y)
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        else:
            X = np.array(X)

        self._classes = np.unique(y)
        if len(self._classes) != 2:
            raise ValueError("Only binary classification supported")

        self._class_to_int = {c: i for i, c in enumerate(self._classes)}
        self._int_to_class = {i: c for c, i in self._class_to_int.items()}

        y_encoded = np.array([self._class_to_int[c] for c in y])
        self._fit_node(X, y_encoded, self._tree)

    def predict(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        else:
            X = np.array(X)

        predicted = []
        for x in X:
            prediction = self._predict_node(x, self._tree)
            predicted.append(self._int_to_class[prediction])
        return np.array(predicted)

    @staticmethod
    def _make_terminal(sub_y, node):
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]
