import numpy as np


class InformationRetrievalMetric:
    """
    Compute the metric for experiment. Here is the detail for all metric.
    https://blog.csdn.net/lsgqjh/article/details/79509169?utm_source=blogxgwz6
    """

    @staticmethod
    def precision(data):
        """
        compute the precision
        :param data: the data must be a tuple.   list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.

        :return: if input data is list, return the precision of this list. if the input data is list of list, return the
        average precision on all list. The second return value is a list of precision for each input.
        """
        if type(data) != list and type(data) != tuple:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]
        if type(data) == tuple:
            if data[1] == 0:
                return 0.0, [0.0]
            else:
                precision = np.mean(data[0])
                return precision, [precision]

        if type(data) == list:
            separate_result = []
            for sub_list, total_num in data:
                if total_num == 0:
                    precision = 0
                else:
                    precision = np.mean(sub_list)

                separate_result.append(precision)
            return np.mean(separate_result), separate_result

    @staticmethod
    def recall(data):
        """
        compute the recall
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.

        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        if type(data) != list and type(data) != tuple:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]
        if type(data) == tuple:
            if data[1] == 0:
                return 0.0, [0.0]
            else:
                recall = np.sum(data[0]) / data[1]
                return recall, [recall]

        if type(data) == list:
            separate_result = []
            for sub_list, total_num in data:
                if total_num == 0:
                    recall = 0
                else:
                    recall = np.sum(sub_list) / total_num

                separate_result.append(recall)
            return np.mean(separate_result), separate_result

    @staticmethod
    def f1(data):
        """
        compute the f1
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.                 ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.

        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        if type(data) != list and type(data) != tuple:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]
        if type(data) == tuple:

            precision, _ = InformationRetrievalMetric.precision(data)
            recall, _ = InformationRetrievalMetric.recall(data)

            if precision == 0 or recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return f1, [f1]

        if type(data) == list:
            separate_result = []
            for single_data in data:
                precision, _ = InformationRetrievalMetric.precision(single_data)
                recall, _ = InformationRetrievalMetric.recall(single_data)

                if precision == 0 or recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                separate_result.append(f1)
            return np.mean(separate_result), separate_result

    @staticmethod
    def mrr(data):
        """
        compute the MRR
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.

        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        if type(data) != list and type(data) != tuple:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]
        if type(data) == tuple:
            (sub_list, total_num) = data
            sub_list = np.array(sub_list)
            if total_num == 0:
                return 0.0, [0.0]
            else:
                ranking_array = 1.0 / (np.array(list(range(len(sub_list)))) + 1)
                mr_np = sub_list * ranking_array

                mr = 0.0
                for team in mr_np:
                    if team > 0:
                        mr = team
                        break
                return mr, [mr]

        if type(data) == list:
            separate_result = []
            for (sub_list, total_num) in data:
                sub_list = np.array(sub_list)

                if total_num == 0:
                    mr = 0.0
                else:
                    ranking_array = 1.0 / (np.array(list(range(len(sub_list)))) + 1)
                    mr_np = sub_list * ranking_array

                    mr = 0.0
                    for team in mr_np:
                        if team > 0:
                            mr = team
                            break

                separate_result.append(mr)
            return np.mean(separate_result), separate_result

    @staticmethod
    def map(data):
        """
        compute the MAP
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.

        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        if type(data) != list and type(data) != tuple:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]
        if type(data) == tuple:
            (sub_list, total_num) = data
            sub_list = np.array(sub_list)
            if total_num == 0:
                return 0.0, [0.0]
            else:
                ranking_array = 1.0 / (np.array(list(range(len(sub_list)))) + 1)

                right_ranking_list = []
                count = 1
                for t in sub_list:
                    if t == 0:
                        right_ranking_list.append(0)
                    else:
                        right_ranking_list.append(count)
                        count += 1

                ap = np.sum(np.array(right_ranking_list) * ranking_array) / total_num
                return ap, [ap]

        if type(data) == list:
            separate_result = []
            for (sub_list, total_num) in data:
                sub_list = np.array(sub_list)

                if total_num == 0:
                    ap = 0.0
                else:
                    ranking_array = 1.0 / (np.array(list(range(len(sub_list)))) + 1)

                    right_ranking_list = []
                    count = 1
                    for t in sub_list:
                        if t == 0:
                            right_ranking_list.append(0)
                        else:
                            right_ranking_list.append(count)
                            count += 1

                    ap = np.sum(np.array(right_ranking_list) * ranking_array) / total_num

                separate_result.append(ap)
            return np.mean(separate_result), separate_result

    @staticmethod
    def fleiss_kappa(data, N, k, n):
        """
        compute fleiss_kappa
        :param data:  N*n matrix, number of people who label task  with specific answer, such as [[0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2]] , there are two task ,each task have 5 optional answers , [0,4]=14 means this task has 14 people label the last answer
        :param N: number of task
        :param k: 0ptional answer for each task
        :param n: number of experimenter
        :return: fleiss_kappa , float
        """
        dataMat = np.mat(data, float)
        oneMat = np.ones((k, 1))
        sum = 0.0
        P0 = 0.0
        for i in range(N):
            temp = 0.0
            for j in range(k):
                sum += dataMat[i, j]
                temp += 1.0 * dataMat[i, j] ** 2
            temp -= n
            temp /= (n - 1) * n
            P0 += temp
        P0 = 1.0 * P0 / N
        ysum = np.sum(dataMat, axis=0)
        # sum all element , not N * n , because one task may be have two or more answers
        if sum < N * n:
            raise Exception("input data is wrong!")
        for i in range(k):
            ysum[0, i] = (ysum[0, i] / sum) ** 2
        Pe = ysum * oneMat * 1.0
        ans = float(P0 - Pe) / (1 - Pe)
        return ans[0, 0]
