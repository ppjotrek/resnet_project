from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import re
from torch.utils.data import Dataset
import wfdb

def prepare_dataset(path:str):

    ecg_data=pd.read_csv(path)

    #Mapowanie subklas do superklas
    diagnostics={"NORM":['NORM','CSD'],
                "STTC":['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL', 'ISCAN' ],
                "MI":['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL'],
                "HYP":['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE', 'SEHYP'],
                "CD":['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB', 'WPW', 'ILBBB', '3AVB', '2AVB'],
                "OTHER":['AFLT', 'AFIB', 'PSVT', 'STACH', 'PVC', 'PACE', 'PAC']
                }

    #Przypisanie labelek
    labels=[]
    discard=[]
    for index in range(ecg_data.shape[0]):
        counter=0
        temp_diag=ecg_data['scp_codes'][index]
        temp_diag=re.sub('{',"",str(temp_diag))
        temp_diag=re.sub('}',"",temp_diag)
        temp_diag=temp_diag.split(',')                 
        len_diag=len(temp_diag)
        for idx in range(len_diag):
            temp_d=temp_diag[idx]
            temp_d=temp_d.split(':')[0]
            temp_d=re.sub(r'[^\w\s]',"",temp_d)
            if temp_d in diagnostics['NORM']:
                label=0
                counter=1
            elif temp_d in diagnostics['STTC']:
                label=1
                counter=1
            elif temp_d in diagnostics['MI']:
                label=2
                counter=1
            elif temp_d in diagnostics['HYP']:
                label=3
                counter=1
            elif temp_d in diagnostics['CD']:
                label=4
                counter=1
            elif temp_d in diagnostics['OTHER']:
                label=5
                counter=1
            else:
                label=100
                
            labels.append(label)
        if counter==0:
            discard.append(index)

    final_labels=[]
    for index in range(len(labels)):
        if labels[index]!=100:
            final_labels.append(labels[index])

    final_data=ecg_data.drop(axis=0,index=discard)
    final_data['labels']=final_labels

    return final_data


class ECG_Data(Dataset):

    def __init__(self, dataframe, path:str):
        self.data = dataframe
        self.path = path


    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self,idx):
        path=self.path+self.data['filename_lr'][idx]
        file_audio=wfdb.rdsamp(path) #czytanie plików danych, próbowałem inaczej ale jak jest do tego biblioteka to czemu nie. Raczej to jest standard przy EKG jak patrzyłem po kagglu
        data=file_audio #zwraca 2 wartości, pierwsza to dane, druga to metadane, metadanych nie wykorzystujemy bo nie ma po co
        data_new=np.array(data[0])
        data_new=np.transpose(data_new,(1,0))
        data_final=data_new[7]
        label=self.data['labels'][idx]
        data_final=torch.Tensor(data_final)
        data_final = data_final[None,:]
        return data_final,label


#Przykładowe użycie

if __name__=="__main__":

    #Najpierw lecisz z prepare_data, bo musisz mieć labelki dla superklas, wrzucasz link do CSV, możesz zostawić ten co jest bo to jest względna
    final_data=prepare_dataset(".\\sample_dataset\\ptbxl_database.csv")

    #split zrobiłem sklearnem bo po co się męczyć
    ECG_train,ECG_test=train_test_split(final_data,test_size=0.2,random_state=42)

    #inaczej nie działa xd
    ECG_train=ECG_train.reset_index()
    ECG_test=ECG_test.reset_index()
    #tu ważne żeby podać ścieżkę do folderu z danymi, ale tylko folderu, bo potem to już się samo sklei
    train_dataset=ECG_Data(ECG_train, ".\\sample_dataset\\")
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64)
    test_dataset=ECG_Data(ECG_test, ".\\sample_dataset\\")
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64)

    print("Train data: ",len(train_dataset))
    print("Test data: ",len(test_dataset))

    #print class distribution
    print("Train class distribution: ")
    print(ECG_train['labels'].value_counts())
    print("Test class distribution: ")
    print(ECG_test['labels'].value_counts())

    print("Printing train dataloader content: ")

    for data, label in train_loader:
        print(data)
        print(label)
        break