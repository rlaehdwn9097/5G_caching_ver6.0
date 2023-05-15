import os
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm

import NetworkEnv.network as nt
import NetworkEnv.content as ct
import NetworkEnv.config as cf
import NetworkEnv.genearal_distribution as gd
# TODO : 여기서는 ContentList 를 Scenario로 반환
# TODO : input --> Content List

class Scenario(object):
    def __init__(self):
        self.contentfile = cf.CONTENT_FILE
        self.generating_method = cf.GEN_METHOD
        self.contentList = self.set_contentList(contentfile = self.contentfile)
        self.titleList = self.set_titleList()
        self.weightList = self.set_weigthList()


        #! for tmp mMTC
        self.mMTC_id = 0

    def set_contentList(self, contentfile):
        contentList = []

        if contentfile == 'mMTC':
            return None
        
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"./data/{contentfile}.csv"))
            
            #rows, _ = df.shape
            keys = list(df.keys())
            for row in tqdm(df.to_numpy(), total = len(df), position=0, leave=True):
                ct_keys = keys
                ct_values = row
                if 'size' not in ct_keys:
                    ct_keys = np.append(ct_keys, ['size'])
                    ct_values = np.append(ct_values, [20])
                if 'requested_date' not in ct_keys:
                    ct_keys = np.append(ct_keys, ['requested_date'])
                    ct_values = np.append(ct_values, [0])
                if 'experied_date' not in ct_keys:
                    ct_keys = np.append(ct_keys, ['expired_date'])
                    ct_values = np.append(ct_values, [0])

                content = ct.Content(ct_keys, ct_values)
                contentList.append(content)
    
        return contentList
    
    def set_titleList(self):
        titleList = []
        for i in range(len(self.contentList)):
            titleList.append(self.contentList[i].id)
            #titleList.append(self.contentList[i].title)
        titleList = set(titleList)
        titleList = list(titleList)
        return titleList
        
    def get_titleList(self):
        return self.titleList
    
    def set_weigthList(self):
        if self.generating_method == 'sequential':
            return None
        elif self.generating_method == 'gaussian':
            return None
        #! mMTC
        elif self.generating_method == 'mMTC':
            return None
        else:
            weightList = getattr(gd, self.generating_method)(self.contentList)

        return weightList
    
    # 우리가 실험하던거
    def requestGenerate_gaussian(self,_day):
        weightList = getattr(gd, self.generating_method)(self, _day)
        choice = random.choices(self.contentList, weights = weightList, k = 1)
        return choice[0]

    def requestGenerate(self, date, _day):
        #! for the zipf
        if self.generating_method == 'zipf':
            choice = random.choices(self.contentList, weights = self.weightList, k = 1)
            #print(choice[0].__dict__)
            return choice[0]
        
        #! for the squential 
        elif self.generating_method == 'sequential': 
            choice = self.contentList[_day]
            return choice
        
        #! for gaussian
        elif self.generating_method == 'gaussian':
            self.weightList = getattr(gd, self.generating_method)(self.contentList, _day)
            choice = random.choices(self.contentList, weights = self.weightList, k = 1)
            #print(choice[0].__dict__)
            return choice[0]
        
        #! tmp mMTC need configuration,,, when loglikelyhood generating is done
        elif self.contentfile == 'mMTC':
            
            valid_list = []
            keys = ['id','temperature','humidty', 'generated_date', 'expiered_date','size']
            # 한 라운드에 3개 생기는 거 --> random X, 어느날은 더 많게 어느 날은 더 적게 count를 정함
            # log likely hood 는 generated date 랑 expiered date and popularity 사용
            # valid list update
            # 최종 valid list 안에서 하나 뽑기.
            for i in range(3):
                values = []
                temperature = random.uniform(10,40)
                humidty = random.uniform(37,45)
                generated_date = date
                expiered_date = generated_date + random.randint(1,3)
                size = 1

                values.append(self.mMTC_id)
                values.append(temperature)
                values.append(humidty)
                values.append(generated_date)
                values.append(expiered_date)
                values.append(size)

                vaild_content = ct.Content(keys, values)
                valid_list.append(vaild_content)
                
            self.mMTC_id += 1
            choice = random.sample(valid_list, 1)
            print(choice[0].__dict__)
            return choice[0]

    def generateRequest(self, _i):
        # TODO : contentList 에서 원하는 generating_method에 따라 data generating
        #weightList = getattr(gd, self.generating_method)(self)
        choice = random.choices(self.contentList, weights = self.weightList, k = 1)
        return choice[0]



