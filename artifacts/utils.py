import pickle
import numpy as np 

class data_predict():
    def __init__(self,data):
        self.data = data

    def load_model(self):
        with open (r'artifacts/boston.pkl','rb') as file:
            self.model = pickle.load(file)


    def predict(self):
        self.load_model()
        
        CRIM = float(self.data['CRIM'])
        ZN = float(self.data['ZN'])
        INDUS=float(self.data['INDUS'])
        CHAS=float(self.data['CHAS'])
        NOX=float(self.data['NOX'])
        RM=float(self.data['RM'])
        AGE=float(self.data['AGE'])
        DIS=float(self.data['DIS'])
        RAD=float(self.data['RAD'])
        TAX=float(self.data['TAX'])
        PTRATIO=float(self.data['PTRATIO'])
        B=float(self.data['B'])
        LSTAT=float(self.data['LSTAT'])

        array  = np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT],ndmin=2)
        result = self.model.predict(array)[0]
        return result

