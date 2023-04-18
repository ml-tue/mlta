import pandas as pd
import os

# TODO Perhaps add functionality to insert previous experience in Metadatabase another way, without GAMA run.

class MetaDataLookupTable:
    def __init__(self, path: str = None):
        """ 'path' is optional to initialize the lookuptable from an existing metadatabase. 
        Should point to folder with three tables:
            'lookup_table_pipelines.csv' with columns 'pipeline' and 'pipeline_id'
            'lookup_table_datasets.csv' with columns 'dataset' and 'dataset_id' """
        self._pipeline_str_to_id: dict = {}
        self._pipeline_id_counter: int = 0
        self._dataset_to_id: dict = {}
        self._dataset_id_counter: int = 0
        self._path = path

        # initialize lookuptable
        if path != None:
            # check if files present in path
            pipe_path = os.path.join(path, "lookup_table_pipelines.csv")
            dataset_path = os.path.join(path, "lookup_table_datasets.csv")
            if not os.path.exists(pipe_path):
                raise ValueError("no 'lookup_table_pipelines.csv' found in specified 'path' ", pipe_path)
            if not os.path.exists(dataset_path):
                raise ValueError("no 'lookup_table_datasets.csv' found in specified 'path' ", dataset_path)

            pipelines_table = pd.read_csv(pipe_path)
            for pipe_entry in zip(pipelines_table["pipeline"], pipelines_table["pipeline_id"]):
                self._pipeline_str_to_id[str(pipe_entry[0])] = int(pipe_entry[1])
                self._pipeline_id_counter = max(int(pipe_entry[1]), self._pipeline_id_counter)

            datasets_table = pd.read_csv(dataset_path)
            for dataset_entry in zip(datasets_table["dataset"], datasets_table["dataset_id"]):
                self._dataset_to_id[str(dataset_entry[0])] = int(dataset_entry[1])
                self._dataset_id_counter = max(int(dataset_entry[1]), self._dataset_id_counter)
            
    def add_pipeline(self, pipeline_str: str):
        # if already exists then do not add
        if not self.pipe_exists(pipeline_str):
            self._pipeline_str_to_id[pipeline_str] = self._pipeline_id_counter
            self._pipeline_id_counter += 1

    def get_pipeline_id(self, pipeline_str: str) -> int:
        return self._pipeline_str_to_id[pipeline_str]

    def pipe_exists(self, pipeline_str: str) -> bool:
        """ Returns boolean indicating whther the pipeline_str already has an associated id, and thus is in the table """
        return pipeline_str in self._pipeline_str_to_id
    
    def list_pipelines(self, by: str) -> list:
        if by == "name":
            return list(self._pipeline_str_to_id.keys())
        elif by == "id":
            return list(self._pipeline_str_to_id.values())
        elif by == "both":
            return list(zip(list(self._pipeline_str_to_id.keys()), list(self._pipeline_str_to_id.values())))

    def add_dataset(self, dataset_str: str):
        while self._dataset_id_counter in list(self._dataset_to_id.values()): #dont accidentally skip a dataset id
            self._dataset_id_counter += 1
        if not self.dataset_exists(dataset_str):
            self._dataset_to_id[dataset_str] = self._dataset_id_counter
            self._dataset_id_counter += 1

    def get_dataset_id(self, dataset_str: str) -> int:
        return self._dataset_to_id[dataset_str]
    
    def dataset_exists(self, dataset_str: str) -> bool:
        """ Returns boolean indicating whther the dataset_str already has an associated id, and thus is in the table """
        return dataset_str in self._dataset_to_id
    
    def list_datasets(self, by: str) -> list:
        if by == "name":
            return list(self._dataset_to_id.keys())
        elif by == "id":
            return list(self._dataset_to_id.values())
        elif by == "both":
            return list(zip(list(self._dataset_to_id.keys()), list(self._dataset_to_id.values())))

    def to_csvs(self, path: str = "") -> None:
        """ Writes the current metadata lookup tables to two csv files at 'path':
            'lookup_table_pipelines.csv'
            'lookup_table_datasets.csv'
        Beware this may overwrite existing files"""

        pipeline_table_path = os.path.join(path, "lookup_table_pipelines.csv")
        dataset_table_path = os.path.join(path, "lookup_table_datasets.csv")

        df_pipelines = pd.DataFrame(data={
            "pipeline": list(self._pipeline_str_to_id.keys()), 
            "pipeline_id": list(self._pipeline_str_to_id.values())
            })
        df_pipelines.to_csv(pipeline_table_path, header=["pipeline", "pipeline_id"])

        df_data = pd.DataFrame(data={
            "dataset": list(self._dataset_to_id.keys()),
            "dataset_id": list(self._dataset_to_id.values())
            })
        df_data.to_csv(dataset_table_path, header=["dataset", "dataset_id"])

    def update_tables(self) -> None:
        "updates the lookuptables that were used for initialization with the current metadatabase"
        if self._path != None:
            self.to_csvs(self._path)
