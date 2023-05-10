import config as cf
import numpy as np
import network as nt

class Content(object):
    def __init__(self, _title, _size, _popularity, _peak_day,_category):
        self.title = _title
        self.size = _size
        self.popularity = _popularity
        self.peak_day =_peak_day
        self.category = _category
    
    def get_title(self):
        return self.title

    def get_popularity(self):
        return self.popularity

class contentStorage(object):
    def __init__(self, _size):
        self.capacity = _size
        self.stored = 0
        self.content_storage=[]
        self.content_req_cnt_list=[]
        
    def abletostore(self,c:Content):
        freeSpace = self.capacity-self.stored
        if(freeSpace>=c.size):
            return 1
        else:
            return 0

    def isfull(self):
        freeSpace = self.capacity-self.stored
        if(freeSpace == 0):
            return 1
        else:
            return 0

    def addContent(self,c:Content):
        self.content_storage.append(c)
        self.content_req_cnt_list.append(1)
        self.stored = self.stored + c.size

    def isstored(self,c:Content):
        if len(self.content_storage)>0:
            for i in self.content_storage:
                if i.title == c.title:
                    return 1
        return 0

    def delContent(self,c:Content):
        newstorage=[]
        newContent_req_cnt_list=[]
        for i in range(len(self.content_storage)):
            if self.content_storage[i] is c:
                self.stored = self.stored-c.size
            else:
                newstorage.append(self.content_storage[i])
                newContent_req_cnt_list.append(self.content_req_cnt_list[i])
        self.content_storage=newstorage
        self.content_req_cnt_list=newContent_req_cnt_list

    def delFirstStored(self):#사용한지 가장 오래된 매체 삭제
        self.stored = self.stored - self.content_storage[0].size 
        self.content_storage=self.content_storage[1:]
        self.content_req_cnt_list=self.content_req_cnt_list[1:]

    



