import os 
import pandas as pd

def get_values(rootDir): 
    li = []
    list_dirs = os.walk(rootDir) 
    for root, dirs, files in list_dirs: 
        name = root.split("/")[1]
        if name not in li and ("net" in name or "Net" in name):
            print("")
            print(name,end=",") 
            li.append(root.split("/")[1])
        for f in files: 
            if f == "training.csv":
                path = os.path.join(root, f)
                df = pd.read_csv(path)
                print(df['val_acc'].max(),end=",")

if __name__ == '__main__':
    get_values("./")
