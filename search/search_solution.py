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
    def __init__(self, data_file='./data/hnsw_0.bin',
                 data_url='https://drive.google.com/file/d/1VTySmcrs-FnuE8lPVAShD4S5qJ_MmJTm/view?usp=sharing') -> None:
        '''
        Creates regestration matrix and passes 
        dictionary. Measures baseline speed on
        a given machine
        '''
        print('reading from Solutions')
        self.data_file = data_file
        self.data_url = data_url

        dim = 512
        max_elements = 2000001
        ef_construction = 200
        M = 16

        self.index = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

    # @profile
    def set_base_from_pickle(self):
        '''
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        '''
        base_file = './data/train_data.pickle'
        base_url = 'https://drive.google.com/file/d/1NfZwLjy0rQ_vGB_nKXjYIu1vm5tgErEg/view?usp=sharing'

        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data') 
            gdown.download(self.data_url, self.data_file, quiet=False)

        if not os.path.isfile(base_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(base_url, base_file, quiet=False)

        # with open(base_file, 'rb') as f:
        #     data = pickle.load(f)
        # self.pass_dict = data['pass']

        self.index.load_index(self.data_file)
        self.ids = {}
        for i in range(self.index.get_current_count()):
            self.ids[i] = i

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
        labels, distances = self.index.knn_query(query, k=10)
        distances = 1 - distances
        return [(self.ids[i], sim) for i, sim in zip(labels[0].tolist(), distances[0].tolist())]

    def insert_base(self, feature: np.array) -> None:
        ## there no inplace concationation in numpy so far. For inplace
        ## concationation operation both array should be contingious in 
        ## memory. For now, let us suffice the naive implementation of insertion
        # self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0)
        # pass
        self.index.add_items(feature)

    def cos_sim(self, query: np.array) -> np.array:
        pass
