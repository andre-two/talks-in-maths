import pandas as pd
import numpy as np

import geopandas as gpd
from geopy.distance import geodesic as GD


# import sys
import os

sp = os.path.abspath(__file__).split('EncodingsHDI.py')[0] + os.sep + "_encoding_data" + os.sep
# sys.path.insert(0, sp)

# def _find_cfg():
#    sp    = []
#    sp[:] = sys.path
#    sp[0] = os.path.abspath(sp[0])
#    sp.insert(1, os.path.expanduser('~/.config/saspy'))
#    sp.insert(0, __file__.rsplit(os.sep+'__init__.py')[0])

#    cfg = 'Not found'

#    for dir in sp:
#       f1 = dir+os.sep+'sascfg_personal.py'
#       if os.path.isfile(f1):
#          cfg = f1
#          break

#    if cfg == 'Not found':
#       f1 =__file__.rsplit('__init__.py')[0]+'sascfg.py'
#       if os.path.isfile(f1):
#          cfg = f1

#    return cfg





class Encoder():
    def __init__(self, object_, type_ = "f"):
        """
        Parameter object_ object: region or regiao or r or vehicle or veiculo or v
        Defines which type of object is going to be detected
        r or region or regiao for regions
        v or vehicle or veiculo for vehicles

        Parameter type_ type: f, freq, frequency, frequencia or s, sev, severity, severidade
        Defines which type of encoding is going to be used
        f for frequency encoding (DEFAULT)
        s for severity encoding

        """

        # Check if type_ is valid
        if type_ != "f" and type_ != "freq" and type_ != "frequency" and type_ != "frequencia" and type_ != "s" and type_ != "sev" and type_ != "severity" and type_ != "severidade":
            raise Exception("Invalid type_, must be frequency or severity")

        # Check if type object is valid
        if object_ != "region" and object_ != "vehicle" and object_ != "r" and object_ != "v" and object_ != "regiao" and object_ != "veiculo":
            raise Exception("Invalid object_, must be region or vehicle")
        
        # Set type_ attribute
        if type_ == "f" or type_ == "freq" or type_ == "frequency" or type_ == "frequencia":
            self.type_ = "f"
        else:
            self.type_ = "s"

        # Set object_ attribute
        if object_ == "r" or object_ == "region" or object_ == "regiao" :
            self.object_ = "r"
        else:
            self.object_ = "v"


        # Load coordinates and map for regions
        # or coordinates and vehicle data for vehicles
        # self.coordinates = None
        self.map_ = None
        self.vehicles = None
        
        # self.coordinates = coordinates.load_coordinates(self.object_)

        if self.object_ == "r":
            self.coordinates = gpd.read_file( sp + "coordinates" + os.sep + "region_coords.shp")
            self.coordinates.rename(columns= {'codRegiaoR' : 'codRegiaoRiscoAuto'}, inplace=True)
            self.coordinates[['centroid', 'LatLong']] =  pd.DataFrame(self.coordinates['geometry'].apply(self.to_centroid).tolist())

            # ID must be object type
            self.coordinates['codRegiaoRiscoAuto'] = self.coordinates['codRegiaoRiscoAuto'].astype(str)

            self.map_ = gpd.read_file(sp + "region_map" + os.sep + "malha3.shp")
            self.map_.rename(columns= {'CodRegiaoR' : 'codRegiaoRiscoAuto'}, inplace=True)


        else:
            self.coordinates = pd.read_csv(sp + "coordinates" + os.sep + "vehicle_coords_v0.csv")
            self.vehicles = pd.read_excel(sp + "vehicle_data" + os.sep + "variaveisMobiSEP2023.xlsx")

        self.Encoding = None
        self.dict_encoding = dict()

        return None
    


    def to_centroid(self, geometry):
        """
        Simple function to extract centroid (center of mass) point as well as coordinates (lot/long) from geometry
        
        Parameters
        ----------
        geometry : Polygon or MultiPolygon
            Shapely geometry object for which we want to calculate centroid

        Returns
        -------
        tuple (centroid: Point, latitude: float, longitude: float)

        """
        return (geometry.centroid, (geometry.centroid.y, geometry.centroid.x))

    def distanceGeo(self, x, y):    
        dist = GD(x, y).km
        return dist

    # def distanceGeoY(self, x):
    #     return self.distanceGeo(self, x, y)

    def get_coordinates(self):
        return self.coordinates

    def get_map(self):
        return self.map_

    def get_vehicles(self):
        return self.vehicles
    
    def MicroExperience(self):
        return None

    def MacroExperience(self, verbose = False):
        return None


    def Credibility(self, lambda_, k, z, std_ = 0 , mean_ = 1):
        """
        cred = sqrt(LAMBDA (num sinistros) / LAMBDA_M)
        LAMBDA_M = (Z / k)^2 * (1 + (STD/MEAN)^2)

        """ 

        STD = std_
        MEAN = mean_


        LAMBDA_M = (z / k)**2 * (1 + (STD/MEAN)**2)

        # CONTINGENCIA PARA GARANTIR QUE O CALCULO PERMANEÇA COERENTE
        if np.isnan(lambda_ ):
            Z_ = 0
        else:
            Z_ = min(np.sqrt(lambda_ / LAMBDA_M),1)

        return Z_




    def Grouping(self, train_data, id, response, weight, r_inf = -1, r_sup = -1, k = 0.1, z = 1.645, cred= 0.5, verbose = False):

        LIST_IDS = self.coordinates[id].to_list()

        LIST_IDS.sort()

        # LIST_IDS = [2609]
        
        Grouped = dict()


        debug = False
        if debug:
            LIST_IDS = ["2609", "3353", "3363"]
        


        
        i = 0
        for r in LIST_IDS :

            if debug:
                print(f"regiao {r}")

            Y = self.coordinates.LatLong[self.coordinates[id] == r]

            region_y = train_data.copy()


            # region_y = region_y.fillna(0)
            if debug:
                print(region_y[region_y.isna().any(axis=1)])


            
            region_y['DIST'] = region_y.LatLong.apply(lambda x: self.distanceGeo(x, Y))
            
            region_y = region_y.sort_values(by = 'DIST').reset_index(drop = True)

            region_y['SUM_RESPONSE'] = region_y[response].cumsum()

            region_y['CREDIBILITY'] = region_y['SUM_RESPONSE'].apply(lambda x: self.Credibility(x, k, z))


            if debug:
                print(region_y[[id, 'DIST', 'SUM_RESPONSE', 'CREDIBILITY']])

            NoNeighbours = min( (region_y['CREDIBILITY']<= cred).sum() + 1 , region_y.shape[0])

            if debug:
                print(f"linhas cred abaixo do limite : {(region_y['CREDIBILITY']<= cred).sum()}")
                print(f"NoNeighbours : {NoNeighbours}")



        
            # NoNeighbours = MacroXP1[ID][0:NoNeighbours].to_list()
            
            # region_y = region_y[region_y['CREDIBILITY'] <= cred]
            region_y = region_y[0:NoNeighbours]

            if debug:
                print("OLHAR AQUI")
                print(region_y[[id, 'DIST', 'SUM_RESPONSE', 'CREDIBILITY']])
                print(region_y['CREDIBILITY']<= cred)

            SUM_RESPONSE = region_y[response].sum()

            if debug:
                print(SUM_RESPONSE)

            SUM_WEIGHT = region_y[weight].sum()

            if debug:
                print(SUM_WEIGHT)


            CRED_Y = self.Credibility(SUM_RESPONSE, k, z)

            if debug:
                print(CRED_Y)

            FREQ_Y = SUM_RESPONSE / SUM_WEIGHT

            if debug:
                print(FREQ_Y)

            Grouped[r] = [SUM_RESPONSE, SUM_WEIGHT, CRED_Y, FREQ_Y]

            i += 1
            if verbose:
                if i % 100 == 0:
                    print(f"Progress: {i : 003}  /  {len(LIST_IDS)}...")

            # print(region_y)
                

            # NoNeighbours = min( (region_y['CREDIBILITY']<= cred).sum() + 1 , region_y[id].count())
            

            # Neighbours = region_y[id][0:NoNeighbours].to_list()
            # Neighbours.sort()
            
            # NumEvents = region_y[response][0:NoNeighbours].sum()
            
            
            # GROUPED.extend(Neighbours)
            # GROUPED = list(set(GROUPED))
        
        return Grouped
    

    
    def fit(self, train_data, valid_data, id, response, weight, r_inf = -1, r_sup = -1, k = 0.1, z = 1.645, cred_micro = 0.4, cred_macro = 0.7, verbose = False ):
        """
        Parameter train_data:   Dataframe with training data
        Parameter valid_data:   Dataframe with validation data
        Parameter id:           String column name containg codRegiaoRiscoAuto
        Parameter response:     String with response column name
        Parameter weight:       String with weight column name
        Parameter r_inf:        Integer with minimum number of regions to be considered in the micro experience
        Parameter r_sup:        Integer with maximum number of regions to be considered in the micro experience
        Parameter cred_micro:   Float with credibility of the micro experience
        Parameter cred_macro:   Float with credibility of the macro experience
        Parameter k:            Accepted difference in percentage between the calculated value and the real one
        Parameter z:            Float with z value for the level of confidence on the frequency / severity / risk ratio 

        """

        if verbose:
            print("Checking parameters...")	


        # Check if r_inf and r_sup are valid
        if r_inf != -1 and r_sup != -1:
            if r_inf > r_sup:
                raise Exception("r_inf must be less than r_sup")
        
        # Check if r_inf and r_sup are valid
        if r_inf != -1 and r_sup == -1:
            raise Exception("r_inf must be used with r_sup")
        
        # Check if r_inf and r_sup are valid
        if r_inf == -1 and r_sup != -1:
            raise Exception("r_sup must be used with r_inf")

        # Check if cred_micro and cred_macro are valid
        if cred_micro > 1 or cred_micro < 0 or cred_macro > 1 or cred_macro < 0:
            raise Exception("cred_micro and cred_macro must be between 0 and 1")

        # Check if train_data and valid_data are valid
        if train_data.shape[0] == 0 or valid_data.shape[0] == 0:
            raise Exception("train_data and valid_data must have at least one row")

        # Check if response and weight are valid 
        if response not in train_data.columns or response not in valid_data.columns:
            raise Exception("response must be a valid column name")

        if weight not in train_data.columns or weight not in valid_data.columns:
            raise Exception("weight must be a valid column name")
        
        if id not in train_data.columns or id not in valid_data.columns:
            raise Exception("weight must be a valid column name")

        # # Check if response and weight are valid
        # if train_data[response].dtype != "float64" or valid_data[response].dtype != "float64":
        #     raise Exception("response must be a float column")

        # if train_data[weight].dtype != "float64" or valid_data[weight].dtype != "float64":
        #     raise Exception("weight must be a float column")

        # Make sure that response and weight are not null
        if train_data[response].isnull().sum() != 0 or valid_data[response].isnull().sum() != 0:
            raise Exception("response must not have null values")

        if train_data[weight].isnull().sum() != 0 or valid_data[weight].isnull().sum() != 0:
            raise Exception("weight must not have null values")
        
        # Check if k and z are valid
        if k < 0 or k > 1:
            raise Exception("k must be between 0 and 1")
        
        if z < 0:
            raise Exception("z must be positive")
        
        if verbose:
            print("Parameters OK\n\n")
            print("Checking data...")

        trainSumm = train_data[[id, response, weight]].groupby(id).sum().reset_index()
        trainSumm['FREQ'] = trainSumm[response]/trainSumm[weight]


        validSumm = valid_data[[id, response, weight]].groupby(id).sum().reset_index()
        validSumm['FREQ'] = validSumm[response]/validSumm[weight]


        self.coordinates.rename(columns= {'codRegiaoRiscoAuto': id}, inplace=True)
        self.map_.rename(columns= {'codRegiaoRiscoAuto': id}, inplace=True)

        XP = self.coordinates.merge(trainSumm, on = id, how = 'left')

        # IMPEDE A CRIACAO DE LINHAS COM NP.NAN NO DATA FRAME O QUE PREJUDICA O AGRUPAMENTO
        XP = XP.fillna({response: 0, weight: 0})



        if verbose:
            print("Data OK\n\n")
            print("Calculating macro experience...")

        FREQ_MACRO = self.Grouping(XP, id, response, weight, r_inf, r_sup, k, z, cred_macro, verbose)

        FREQ_MACRO = pd.DataFrame(FREQ_MACRO).T
        FREQ_MACRO = FREQ_MACRO.reset_index()
        FREQ_MACRO.columns = [id, 'SUM_RESPONSE_0', 'SUM_WEIGHT_0', 'CRED_0', 'FREQ_0']

        if verbose:
            print("Macro experience DONE\n\n")
            print("Calculating micro experience...")

        FREQ_MICRO = self.Grouping(XP, id, response, weight, r_inf, r_sup, k, z, cred_micro, verbose)

        FREQ_MICRO = pd.DataFrame(FREQ_MICRO).T
        FREQ_MICRO = FREQ_MICRO.reset_index()
        FREQ_MICRO.columns = [id, 'SUM_RESPONSE_1', 'SUM_WEIGHT_1', 'CRED_1', 'FREQ_1']

        if verbose: 
            print("Micro experience DONE\n\n")
            print("Calculating encoding...")


        FREQ_HAT = FREQ_MACRO.merge(FREQ_MICRO, on = id, how = 'left')
        
        def freq_hat(f0, c0, f1, c1):
            freq_hat = f0 + (f1 - f0) * (c1 / c0)

            return freq_hat

        FREQ_HAT['freq_hat'] = FREQ_HAT.apply(lambda x: freq_hat(x['FREQ_0'], x['CRED_0'], x['FREQ_1'], x['CRED_1']), axis = 1)



        if self.object_ == "r":
            FREQ_HAT['EncodingRegiao'] = np.log(FREQ_HAT['freq_hat'] + 0.00001)

            # normalize EncodingRegiao from 1 to 1000
            FREQ_HAT['EncodingRegiao'] = FREQ_HAT['EncodingRegiao'] - FREQ_HAT['EncodingRegiao'].min()
            FREQ_HAT['EncodingRegiao'] = FREQ_HAT['EncodingRegiao'] / FREQ_HAT['EncodingRegiao'].max()
            FREQ_HAT['EncodingRegiao'] = FREQ_HAT['EncodingRegiao'] * 999 + 1
            FREQ_HAT['EncodingRegiao'] = FREQ_HAT['EncodingRegiao'].astype(int)

            # CARREGA O MAPA COM O ENCODING PARA VISUALIZACAO
            self.map_ = self.map_.merge(FREQ_HAT[[id, 'EncodingRegiao']], on = id, how = 'left')

        else:
            FREQ_HAT['EncodingVeiculo'] = np.log(FREQ_HAT['freq_hat'] + 0.00001)

            # normalize EncodingVeiculo from 1 to 1000
            FREQ_HAT['EncodingVeiculo'] = FREQ_HAT['EncodingVeiculo'] - FREQ_HAT['EncodingVeiculo'].min()
            FREQ_HAT['EncodingVeiculo'] = FREQ_HAT['EncodingVeiculo'] / FREQ_HAT['EncodingVeiculo'].max()
            FREQ_HAT['EncodingVeiculo'] = FREQ_HAT['EncodingVeiculo'] * 999 + 1
            FREQ_HAT['EncodingVeiculo'] = FREQ_HAT['EncodingVeiculo'].astype(int)

            

        
        # Score = FREQ_HAT[[id, 'freq_hat']].copy()

        if verbose:
            print("Encoding DONE\n\n")  

        self.Encoding = FREQ_HAT.copy()

        self.dict_encoding = dict(zip(self.Encoding[id], self.Encoding['EncodingRegiao']))
        self.dict_freq_hat = dict(zip(self.Encoding[id], self.Encoding['freq_hat']))

        self.Valid_AgSE, self.Valid_MSE = self._eval(validSumm, id, weight)
        self.Train_AgSE, self.Train_MSE = self._eval(trainSumm, id, weight)

        print(f"Train Aggregated SE: {self.Train_AgSE}")
        print(f"Train MSE: {self.Train_MSE}")
        print(f"Valid Aggregated SE: {self.Valid_AgSE}")
        print(f"Valid MSE: {self.Valid_MSE}")

        return FREQ_HAT
    
    def predict(self, df, id):
        """
        Parameter df: Dataframe with data to be encoded
        Parameter id: String column name containg codRegiaoRiscoAuto
        """

        if self.Encoding is None:
            raise Exception("Encoding is not defined, please run fit method first")

        if id not in df.columns:
            raise Exception("id must be a valid column name")

        # if self.object_ == "r":
        #     df['EncodingRegiao'] = df[id].map(self.dict_encoding)
        # else:
        #     df['EncodingVeiculo'] = df[id].map(self.dict_encoding)

        # df['EncodingRegiao'] = df[id].map(self.dict_encoding)

        return df[id].map(self.dict_encoding)
    

    def _eval(self, df, id, weight): 
        """
        Parameter df: Dataframe with data to be encoded
        """
        
        FREQ_HAT= df[id].map(self.dict_freq_hat)
        FREQ_REAL = df['FREQ']
        WEIGHT = df[weight]

        # aggregated SSE / MSE
        return ((FREQ_HAT- FREQ_REAL)**2).sum(), ((WEIGHT*(FREQ_HAT- FREQ_REAL)**2).sum() / WEIGHT.sum())
    

