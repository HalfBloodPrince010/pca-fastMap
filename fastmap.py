# Team Members:

# Prashanth Srikanth Pujar [USC ID: 5616456770]
# Srivathsa Sripathi Rao[USC ID: 4140852851]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from copy import deepcopy


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
            for i in range(self.N):
                ax.text(data['x'].loc[i], data['y'].loc[i], str(data['words'].loc[i]), style='italic', fontsize=12)
            ax.axis([0, 14, 0, 9])
            plt.show()
            print("Final embedded Coordinates of words:")
            res = np.array([self.result[0], self.result[1]])
            resdataframe = pd.DataFrame(res.T)
            resdataframe.columns = ["final_x", "final_y"]
            print(resdataframe)
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
        count = 5
        while count > 0:
            object2 = self.furthest_points(object_1, columns)
            temporary = self.furthest_points(object2, columns)
            if temporary == object_1:
                break
            else:
                object_1 = object2
            count -= 1
        print("Pivots Selected:{},{}".format(object_1, object2))
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
    dataframe = pd.read_csv('fastmap-data.txt', names=["obj1", "obj2", "distances"], delimiter="\t")
    # converting pandas dataframe to numpy array
    dataRV = dataframe
    with open('fastmap-wordlist.txt', 'r') as fd:
        words = fd.read().splitlines()
    obj = Fastmap(dataRV, 2, 10, words)
    obj.train_fastmap(2, 0)

