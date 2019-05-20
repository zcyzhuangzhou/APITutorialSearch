import numpy as np


class VectorUtil:
    @staticmethod
    def get_weight_mean_vec(vector_list, weight_list):
        """
        get the average word2vec for list of str
        :param weight_list:
        :param vector_list:
        :return: np.array()
        """
        # todo: add a empty zero vectors result.
        # weight_sum = np.sum(weight_list)
        # normal_weight_list = []
        # for w in weight_list:
        #     normal_weight_list.append(w / weight_sum)
        # weight_list=normal_weight_list

        x = np.matrix(vector_list)
        avg_vector = np.average(x, axis=0, weights=weight_list)
        avg_vector = avg_vector.getA()[0]
        return avg_vector

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        index_2_key_map = {}

        index = 0

        count_list = []
        for key, count in number_dict.items():
            index_2_key_map[index] = key
            count_list.append(count)
            index = index + 1

        a = np.array(count_list)
        ## smooth, in case the divide by zero error
        a = np.log((total_num + 1) / (a + 1))
        result = {}

        for index, w in enumerate(a):
            key = index_2_key_map[index]
            result[key] = w

        return result
