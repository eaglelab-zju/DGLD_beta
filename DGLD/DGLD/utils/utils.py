import numpy as np
import torch
import random
import os.path as osp

def print_shape(*a):
    for t in a:
        print(t.shape)
        
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class ExpRecord():
    def __init__(self, filepath='result.csv'):
        self.filepath = filepath
        if osp.exists(self.filepath):
            self.record = self.load_record()
        else:
            self.record = None
    def init_record(self, dict_record):
        pass

    def add_record(self, dict_record):
        print(dict_record)
        if not self.record:
            self.record = {k:[v] for k, v in dict_record.items()}
        else:
            for k in dict_record:
                if k not in self.record:
                    self.record[k] = [''] * (len(self.record[list(self.record.keys())[0]])-1)
                self.record[k].append(dict_record[k])
        self.save_record()

    def save_record(self):
        pd.DataFrame(self.record).to_csv(self.filepath, index=None)
    
    def load_record(self):
        csv_file = pd.read_csv(self.filepath)
        self.record = {k:list(csv_file[k]) for k in csv_file.columns}
        return self.record 
