import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from copy import deepcopy


def levenshtein_distance(word_1, word_2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if word_1[m-1] == word_2[n-1]:
        return levenshtein_distance(word_1, word_2, m-1, n-1)
    return 1 + min(levenshtein_distance(word_1, word_2, m, n-1), levenshtein_distance(word_1, word_2, m-1, n), levenshtein_distance(word_1, word_2, m-1, n-1))


class Fastmap:
    def __init__(self, data, k, n_objects, wordlist):
        self.data = data
        self.reduced_dimension = k
        self.N = n_objects
        # self.result = np.zeros((self.N, k))
        self.result = [[] for _ in range(self.reduced_dimension)]
        self.word_list = wordlist

    def train_fastmap(self, iterations, columns):
        if iterations <= 0:
            # Creating a dataframe to plot the graph
            graph_list = deepcopy(self.result)
            graph_list.append(self.word_list)
            data = pd.DataFrame(graph_list)
            data = data.transpose()
            # Columns of the data frame
            data.columns = ['x', 'y', 'words']
            # Adding the heading
            fig = plt.figure()
            fig.suptitle('Word Mapping', fontsize=16, fontweight='bold')
            # Adding subplot
            ax = fig.add_subplot(111)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            for index in range(self.N):
                ax.text(data['x'].loc[index], data['y'].loc[index], str(data['words'].loc[index]), style='italic', fontsize=12)
            ax.axis([0, 15, 0, 15])
            plt.show()
            return
        fobject1, fobject2 = self.pivots(columns)
        self.projections(fobject1, fobject2, columns)
        self.train_fastmap(iterations-1, columns+1)

    def distance(self, object_a, object_b, column):
        obj_a = min(object_a, object_b)
        obj_b = max(object_a, object_b)
        if column == 0:
            if (obj_a - obj_b) == 0:
                return 0
            # min_obj = min(object_a, object_b)
            distance_row = self.data.loc[self.data['obj1'] == obj_a].loc[self.data['obj2'] == obj_b]
            # index  obj1  obj2  distances
            #   5     1      7      4
            distance = distance_row['distances'].values[0]  # values returns a list of numbers of the given feature
            # print("distance between", obj_a, "and", obj_b, "-", distance)
            return distance
        else:
            distances = self.distance(object_a, object_b, column-1)
            # xi-xj
            resultant = self.result[column-1][obj_a-1] - self.result[column-1][obj_b-1]
            return (distances**2-resultant**2)**0.5

    def furthest_points(self, ob1, columns):
        ob2 = None
        max_distance = float('-inf')
        for i in range(1, self.N+1):
            dist = self.distance(ob1, i, columns)
            if dist > max_distance:
                max_distance = dist
                ob2 = i
        # print("Max Distance", max_distance)
        return ob2

    def pivots(self, columns):
        object_1 = random.randint(1, self.N)
        print("Object1:", object_1)
        count = 5
        while count > 0:
            object2 = self.furthest_points(object_1, columns)
            print("Object 2:", object2)
            temporary = self.furthest_points(object2, columns)
            if temporary == object_1:
                break
            else:
                object_1 = object2
            count -= 1
            print("====================END OF ITERATION==============================")
        print("Pivots", object_1, object2)
        return object_1, object2

    def projections(self, fobj1, fobj2, columns):
        """
        xi = (D(oA,Ob)**2 + D(oA,oi)**2 - D(oi,oB))/D(oA,oB)
        :return:
        """
        for i in range(1, self.N+1):
            xi = (self.distance(fobj1, fobj2, columns)**2 + self.distance(fobj1, i, columns)**2 - self.distance(i, fobj2, columns)**2)/(2*self.distance(fobj1, fobj2, columns))
            self.result[columns].append(xi)


if __name__ == "__main__":
    # Process the Data
    print("Processing the data")
    dataframe = pd.read_csv('fastmap-data.txt', names=["obj1", "obj2", "distances"], delimiter="\t")
    rand_words = []
    n = int(input("Enter the number of words:"))
    for i in range(n):
        print(i+1, "th word")
        word = input("Enter:")
        rand_words.append(str(word))
    print(rand_words)
    objlist1 = []
    objlist2 = []
    distance_list = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            objlist1.append(i)
            objlist2.append(j)
            distance_list.append(levenshtein_distance(rand_words[i-1], rand_words[j-1], len(rand_words[i-1]), len(rand_words[j-1])))
    datalist = list()
    datalist.append(objlist1)
    datalist.append(objlist2)
    datalist.append(distance_list)
    df = pd.DataFrame(datalist)
    df = df.transpose()
    # Columns of the data frame
    df.columns = ['obj1', 'obj2', 'distances']
    print(df)
    obj = Fastmap(df, 2, len(rand_words), rand_words)
    obj.train_fastmap(2, 0)

