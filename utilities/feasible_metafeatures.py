from typing import List
import os
import arff
from pymfe.mfe import MFE
from contextlib import contextmanager
import threading
import _thread

def get_feasible_metafeatures(metadatabase_path: str = "metadatabase_openml18cc", 
                              max_time: int = 5,
                              meta_features: List[str] = None,
                              verbosity: int = 1) -> List[str]:
    """Returns which pyMFE meta-features are feasible for all datasets stored in the metadatabase.
    That is they run within some time limit on each and everyone of the datasets.
    
    Arguments
    ---------
    metadatabase_path: str, 
        Specifies the path having the datasets for which to compute the applicable meta features.
        Assumes each dataset to be in arff format and for the last feature to be the target.
    max_time: int,
        Maximum time in seconds which a meta feature is allowed on any of the datasets
    meta_features: List[str],
        List of meta-features to try out. If default(`None`), then try out all features of types:
            "complexity", "general", "info-theory", "landmarking", "model-based", "statistical"
    verbosity: int,
        if set to 1 shows which features are removed and why.

    Returns
    -------
    List of strings which can be used to indicate features in pyMFE, subset of `meta-features`
    """
    # define some helper functions/classes
    def isNaN(num):
        if float('-inf') < float(num) < float('inf'):
            return False 
        else:
            return True
        
    class TimeoutException(Exception):
        def __init__(self, msg=''):
            self.msg = msg

    @contextmanager
    def time_limit(seconds, msg=''):
        timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutException("Timed out for operation {}".format(msg))
        finally:
            # if the action ends in specified time, timer is canceled
            timer.cancel()
    
    # start actual execution
    potentially_feasible_features = meta_features # keep track of which meta features to consider
    if meta_features == None:
        groups =["complexity", "general", "info-theory", "landmarking", "model-based", "statistical"]
        potentially_feasible_features = list(MFE().valid_metafeatures(groups=groups))
    database_dir = os.path.join(metadatabase_path, "datasets")

    for dataset in os.listdir(database_dir):
        if verbosity == 1:
            print("current dataset: {}".format(dataset))
        data = arff.load(open(os.path.join(database_dir, dataset), "r"))["data"]
        X = [i[:-1] for i in data] # assumes last feature is the target
        y = [i[-1] for i in data] # assumes last feature is the target

        # try meta-features on this dataset, which are kept up to date.
        for feature in potentially_feasible_features:
            ft = None
            is_feasible = True
            try:
                with time_limit(max_time):
                    allmf_summary = ["min", "max", "mean"]
                    # Extract default measures
                    mfe = MFE(groups="all", features=[feature], summary=allmf_summary, suppress_warnings=True)
                    mfe.fit(X, y)
                    ft = mfe.extract()
            except TimeoutException as e:
                is_feasible = False
                if verbosity == 1:
                    print("Timed out for feature: {}".format(feature))
            if ft != None: # otherwise time-out happened and cannot perform NaN check
                for feature_value in ft[1]:
                    if isNaN(feature_value):
                        is_feasible = False
                        if verbosity == 1:
                            print("Feature contains a NaN value: {}".format(feature))
            if not is_feasible:
                potentially_feasible_features.remove(feature)
    
    # all infeasible features have been removed by now
    return potentially_feasible_features