import pandas as pd
import numpy as np
import os
from tqdm import tqdm
#import seaborn as sns
import pickle

print('\n\n')
raw = os.path.join(os.getcwd(), 'raw')
print('Raw file folder:', raw)
binary = os.path.join(os.getcwd(), 'data', '3w.pkl')
print ('Output binary:', binary)
if os.path.exists(binary):
    os.remove(binary)
print()


events_names = {0: 'Normal',
                1: 'Abrupt Increase of BSW',
                2: 'Spurious Closure of DHSV',
                3: 'Severe Slugging',
                4: 'Flow Instability',
                5: 'Rapid Productivity Loss',
                6: 'Quick Restriction in PCK',
                7: 'Scaling in PCK',
                8: 'Hydrate in Production Line',
                }


variables = ['P-PDG',
             'P-TPT',
             'T-TPT',
             'P-MON-CKP',
             'T-JUS-CKP',
             'P-JUS-CKGL',
             'T-JUS-CKGL',
             'QGL',
            ]




with os.scandir(raw) as dirs, open(binary, 'ab+') as output:
    for directory in dirs:
        raw_data = {}
        if directory.is_dir():
            try:
                entry = int(directory.name)
                path = directory.path
                print(entry, path)
                df_list = []
                with os.scandir(path) as files:
                    for file in tqdm(files):
                        if file.is_file():
                            df = pd.read_csv(file.path)
                            if 'WELL' in file.name:
                                df['type'] = 'EXP'
                                df['case'] = file.name.split('_')[0]
                            else:
                                if 'SIMULATED' in file.name:
                                    df['type'] = 'SIM'
                                else:
                                    df['type'] = 'DRW'
                                df['case'] = file.name[:-4]
                            df.timestamp = pd.to_datetime(df.timestamp)
                            df.set_index('timestamp', inplace=True)
                            df_list.append(df)
                df = pd.concat(df_list)
                del df_list
                df['class'] = df['class'].map(lambda x: 100 - x if x >=100 else  x)
                for case in tqdm(np.unique(df.case)):
                    raw_data[case] = df[df.case==case].drop(columns='case')
                pickle.dump((entry, raw_data), output)
                del df
            except Exception as e:
                raise e
        del raw_data
