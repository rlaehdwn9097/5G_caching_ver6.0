import csv
import config as cf
import random
import content as ct
from time import strftime, localtime, time

class mMTC(object):

    def __init__(self):
        self.nb_data = cf.MAX_ROUNDS
        self.contentList = self.set_mMTC_scenario()

    def generate_days(self):
        total_day = 7*cf.TOTAL_PRIOD
        days = random.choices(range(total_day), k=cf.MAX_ROUNDS)
        days.sort()
        self.days = days

    def set_mMTC_scenario(self):
        contentList = []
        for i in range(self.nb_data):

            keys = ['id','temperature','humidty', 'generated_date', 'expiered_day','size']
            values = []
            
            id = i
            temperature = random.uniform(10,40)
            humidty = random.uniform(37,45)
            light = random.uniform(45,450)
            voltage = random.uniform(2,3)

            generated_date = 
            expiered_date = 
            size = 1

            values.append(id)
            values.append(temperature)
            values.append(humidty)
            values.append(light)
            values.append(voltage)
            values.append(generated_date)
            values.append(expiered_date)
            values.append(size)

            content = ct.Content(keys, values)
            contentList.append(content)
        
        return contentList
        

mMTC()
