from search.search_baseline import SearchBase
from search.search_solution import SearchSolution

if __name__ == "__main__":

    ## Download the data and set speed base 
    searcher = SearchBase()
    searcher.set_base_from_pickle()
