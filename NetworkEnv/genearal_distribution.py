import numpy as np
from scipy.special import zeta 

def gaussian(contentList,_day):
    weightList = []
    for i in range(len(contentList)):
        howAfter = abs(contentList[i].peak_day-_day)
        if howAfter>4:
            howAfter=7-howAfter
        weight = round(gaussian_function(0,howAfter,2)*contentList[i].popularity,4)
        weightList.append(weight)

    return weightList

def gaussian_function(x,mean,sigma):
    return(1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mean)**2/(2*sigma**2))

# n : number of content
# k : rank
# a : Distribution parameter
def zipf(contentList):
    weightList = []
    n = len(contentList)
    a = 1 # or 0 for testing
    result = 0
    for i in range(1,n+1):
        result += i**-a
    for k in range(1, n+1):
        weightList.append((1/k**a)/result)
    
    return weightList

def union(senario):
    weightList = []
    for i in range(len(senario.contentList)):
        weight = 1/len(senario.contentList)
        weightList.append(weight)
    return weightList


'''
def requestGenerate_Liklihood(self, _day):
    titleList = []
    weightList = []
    for i in range(len(self.contentList)):
        titleList.append(self.contentList[i].title)
        howAfter = abs(self.contentList[i].generated_day - _day)
        if howAfter>4:
            howAfter=7-howAfter

        howAfter_std = math.std(howAfter)  # 표준편차 구하기
        weight = round(likelihood(0, howAfter, howAfter_std)*self.contentList[i].popularity,4)

        weightList.append(weight)
    choice = random.choices(titleList, weights = weightList, k = 1)
    for i in self.contentList:
        if i.title == choice[0]:
            return 1

def likelihood(x, mean, std):
    return (1 / math.sqrt(2*math.pi) * math.pow(std, 2)) * np.exp(-(np.power(x - mean, 2) / (2*math.pow(std, 2))))
'''