import pickle, os, time, gdown
import numpy as np
import hnswlib
from copy import copy

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg 
from .search import Base


class SearchSolution(Base):
    ''' SearchBase class implements 
    search through the database to find matching
    vector for query vector. It measures
    search speed and assign a score based
    on the search time and correctness of the
    search 
    '''
    # @profile
    def __init__(self, data_file='./data/list_hnsw_5000.bin',
                 data_url="https://drive.google.com/uc?id=1-0cL9FZVeU-n-liiNu2TMVnEOPYu3W_8") -> None:
        '''
        Creates regestration matrix and passes
        dictionary. Measures baseline speed on
        a given machine
        '''
        print('reading from Solutions')
        self.data_file = data_file
        self.data_url = data_url
        self.pass_dict = None
        self.dim = 512
        self.part_step = 5000
        self.max_elements = 5100
        self.ef_construction = 200
        self.M = 16

        self.reg_matrix = []
        self.set_base_from_pickle()

    # @profile
    def set_base_from_pickle(self):
        '''
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        '''
        base_file = './data/train_data.pickle'
        base_url = "https://drive.google.com/uc?id=1NfZwLjy0rQ_vGB_nKXjYIu1vm5tgErEg"
        ""
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(self.data_url, self.data_file, quiet=False)

        if not os.path.isfile(base_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(base_url, base_file, quiet=False)

        with open(base_file, 'rb') as f:
            data = pickle.load(f)
        self.pass_dict = data['pass']

        with open(self.data_file, 'rb') as f:
            graphs = pickle.load(f)
        self.reg_matrix = graphs

    # @profile
    def cal_base_speed(self, base_speed_path='./base_speed.pickle') -> float:
        '''
        Validates baseline and improved searh
        Return:
                metric : float - score for search
        '''

        samples = cfg.samples
        N, C, C_time, T_base = 0, 0, 0, 0
        for i, tup in enumerate(tqdm(self.pass_dict.items(), total=samples)):

            idx, passes = tup
            for q  in passes:
                t0 = time.time()
                c_output = self.search(query=q)
                t1 = time.time()
                T_base += (t1 - t0)

                C_set = [True for tup in c_output if tup[0] == idx]
                if len(C_set):
                    C += 1
                    C_time += (t1 - t0)
                N += 1

            if i > samples:
                break

        base_speed = T_base / N
        print(f"Solution Line Speed: {base_speed}")
        print(f"Solution Line Accuracy: {C / N * 100}")
        with open(base_speed_path, 'wb') as f:
            pickle.dump(base_speed, f)

    # @profile
    def search(self, query: np.array) -> List[Tuple]:
        '''
        Baseline search algorithm.
        Uses simple matrix multiplication
        on normalized feature of face images

        Arguments:
            query : np.array - 1x512
        Return:
            List[Tuple] - indicies of search, similarity
        '''
        all_distances = np.array([])
        all_labels = np.array([])
        for graph_idx, graph_hnws in enumerate(self.reg_matrix):
            try:
                # print(graph_idx, graph_hnws.get_current_count())
                if graph_hnws.get_current_count() < 5:
                    continue
                labels, distances = graph_hnws.knn_query(query, k=5)
                labels = labels + graph_idx * self.part_step
                # distances = 1 - distances
                # all_labels += labels[0].tolist()
                # all_distances += distances[0].tolist()
                all_labels = np.concatenate((all_labels, labels[0]), axis=None)
                all_distances = np.concatenate((all_distances, distances[0]), axis=None)
            except Exception as e:
                print(f"graph_idx:{graph_idx}", e)
        all_labels = all_labels.tolist()
        all_distances = all_distances.tolist()
        result = list(zip(all_labels, all_distances))
        return result

    def insert_base(self, feature: np.array) -> None:
        ## there no inplace concationation in numpy so far. For inplace
        ## concationation operation both array should be contingious in
        ## memory. For now, let us suffice the naive implementation of insertion
        # self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0)
        graph_hnws = self.reg_matrix[-1]
        if self.part_step-graph_hnws.get_current_count() > feature.size[0]:
            graph_hnws.add_items(feature)
        else:
            part_0 = self.part_step-graph_hnws.get_current_count()
            graph_hnws.add_items(feature[0:part_0])
            for graph_idx, start_idx in enumerate(range(part_0, feature.size[0], self.part_step)):
                part_feature = feature[start_idx:start_idx + self.part_step]
                p = hnswlib.Index(space='cosine', dim=self.dim)  # possible options are l2, cosine or ip
                p.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
                p.add_items(part_feature)
                self.reg_matrix.append(copy(p))
                del p

    def cos_sim(self, query: np.array) -> np.array:
        pass
