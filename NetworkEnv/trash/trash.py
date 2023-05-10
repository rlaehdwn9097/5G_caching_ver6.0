def act(self, path, requested_content, action):

        # self.round_day 기준으로 3일 뒤까지 requested 된 횟수를 가지고 비교 후 caching 할지 말지 정함
        
        # self.roundRequestDictionary : [일, 월, 화, 수, 목, 금, 토] : 각 날짜 별 request 횟수 저장함
        # self.roundRequestDictionary = { "title_1" = [0,2,0,14,20,3,4], ... , "title_N" = [30,5,0,0,0,3,8]}
        # round_day 가 0이면 0,1,2 즉 일,월,화 의 requested 된 횟수의 합을 가지고 비교함
        date = self.days[self.round_nb]
        round_day =  date % 7

        requested_content:ct.Content = requested_content
        #!
        requested_content_title = requested_content.id
        tmp_round_request = 0
        roundRequested_cnt_list = []
        path = path

        requested_content_round_cnt = self.roundRequestDictionary[requested_content_title][round_day] + self.roundRequestDictionary[requested_content_title][(round_day+1)%7] + self.roundRequestDictionary[requested_content_title][(round_day+2)%7] + self.roundRequestDictionary[requested_content_title][(round_day+3)%7]
        #print("\n\n")
        #print("requested content : {}".format(requested_content_title))
        #print("round_day : {}".format(round_day))
        #print(requested_content_title + " : " + str(self.roundRequestDictionary[requested_content_title]))
        #print("cnt : {}".format(requested_content_round_cnt))

        # !MicroBS 에 저장 ---> 꽉차있으면 앞에꺼(가장 업데이트가 안된 컨텐츠) 하나 지움
        # !삭제는 추후 Gain 에 의해서 delete

        # !제일 덜 나온 친구랑 새로 들어올 놈이랑 비교해서 넣을지 말지 
        # !popularity 비교
        if action == 0:
            
            # get_c_node 에 쓰일 변수
            self.stored_type = 0
            self.stored_nodeID = path[1]

            # 저장이 되어 있나? -> 저장할 공간이 있나? -> 1. 저장. / 2. 삭제 후 저장.
            if self.search(self.microBSList, path[1]).storage.isstored(requested_content) != 1:

                if self.search(self.microBSList, path[1]).storage.abletostore(requested_content):
                    self.search(self.microBSList, path[1]).storage.addContent(requested_content)
                    requested_content.update_validity(date)
                    self.cache_changed = 1
                    self.cache_changed_cnt += 1                    

                else:
                    for i in range(len(self.search(self.microBSList, path[1]).storage.content_storage)):
                        storage_content_id = self.search(self.microBSList, path[1]).storage.content_storage[i].id
                        tmp_round_request = self.roundRequestDictionary[storage_content_id][round_day] + self.roundRequestDictionary[storage_content_id][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        
                        #print(str(storage_content_id) + " : " +str(self.roundRequestDictionary[storage_content_id]) + ' ' + str(tmp_round_request))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))
                    #print("requested_content_round_cnt : ", requested_content_round_cnt)
                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.search(self.microBSList, path[1]).storage.content_storage[min_index]


                        #print("del_content : {}".format(del_content.__dict__))
                        #print("바뀌기 전")
                        
                        for j in range(len(self.search(self.microBSList, path[1]).storage.content_storage)):
                            print(self.search(self.microBSList, path[1]).storage.content_storage[j].__dict__)

                        self.search(self.microBSList, path[1]).storage.delContent(del_content)
                        self.search(self.microBSList, path[1]).storage.addContent(requested_content)
                        requested_content.update_validity(date)

                        #print("바뀐 후")
                        #for j in range(len(self.search(self.microBSList, path[1]).storage.content_storage)):
                        #    print(self.search(self.microBSList, path[1]).storage.content_storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1
                        #print("===========================MBS Cache changed===========================")


        # BS 에 저장 ---> 꽉차있으면 앞에꺼 하나 지움
        elif action == 1:
            
            # get_c_node 에 쓰일 변수
            self.stored_type = 1
            self.stored_nodeID = path[2]

            if self.search(self.BSList,path[2]).storage.isstored(requested_content) != 1:
                if self.search(self.BSList,path[2]).storage.abletostore(requested_content):
                    self.search(self.BSList, path[2]).storage.addContent(requested_content)
                    requested_content.update_validity(date)
                    self.cache_changed = 1
                    self.cache_changed_cnt += 1

                else:
                    for i in range(len(self.search(self.BSList, path[2]).storage.content_storage)):
                        storage_content_id = self.search(self.BSList, path[2]).storage.content_storage[i].id
                        tmp_round_request = self.roundRequestDictionary[storage_content_id][round_day] + self.roundRequestDictionary[storage_content_id][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        #print(str(storage_content_id) + " : " +str(self.roundRequestDictionary[storage_content_id]))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))

                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.search(self.BSList, path[2]).storage.content_storage[min_index]
                        
                        #print("del_content : {}".format(del_content.__dict__))
                        #print("바뀌기 전")
                        #for j in range(len(self.BSList[path[2]].storage.content_storage)):
                        #    print(self.BSList[path[2]].storage.content_storage[j].__dict__)

                        self.search(self.BSList, path[2]).storage.delContent(del_content)
                        self.search(self.BSList, path[2]).storage.addContent(requested_content)
                        requested_content.update_validity(date)
                        #print("바뀐 후")
                        #for j in range(len(self.BSList[path[2]].storage.content_storage)):
                        #    print(self.BSList[path[2]].storage.content_storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1
                        #print("===========================BS Cache changed===========================")



        # DataCenter 에 저장 ---> 꽉차있으면 앞에꺼 하나 지움
        elif action == 2:
            #print("its action 2")
            # get_c_node 에 쓰일 변수
            self.stored_type = 2
            self.stored_nodeID = path[3]

            if self.dataCenter.storage.isstored(requested_content) != 1:
                if self.dataCenter.storage.abletostore(requested_content):
                    #print("able to store 인데",len(self.dataCenter.storage.content_storage))
                    #print(len(self.dataCenter.storage.content_storage),self.dataCenter.storage.content_storage)
                    self.dataCenter.storage.addContent(requested_content)
                    requested_content.update_validity(date)
                    #print(len(self.dataCenter.storage.content_storage),self.dataCenter.storage.content_storage)
                    self.cache_changed = 1
                    self.cache_changed_cnt += 1
                    
                else:
                    
                    for i in range(len(self.dataCenter.storage.content_storage)):
                        storage_content_id = self.dataCenter.storage.content_storage[i].id
                        tmp_round_request = self.roundRequestDictionary[storage_content_id][round_day] + self.roundRequestDictionary[storage_content_id][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_id][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        #print(str(storage_content_id) + " : " +str(self.roundRequestDictionary[storage_content_id]))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))

                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.dataCenter.storage.content_storage[min_index]

                        #print("del_content : {}".format(del_content.__dict__))
                        #print("바뀌기 전")
                        #for j in range(len(self.dataCenter.storage.content_storage)):
                        #    print(self.dataCenter.storage.content_storage[j].__dict__)
                        #print("delete")
                        #print(len(self.dataCenter.storage.content_storage),self.dataCenter.storage.content_storage)
                        self.dataCenter.storage.delContent(del_content)
                        #print(len(self.dataCenter.storage.content_storage),self.dataCenter.storage.content_storage)
                        self.dataCenter.storage.addContent(requested_content)
                        requested_content.update_validity(date)
                        #print(len(self.dataCenter.storage.content_storage),self.dataCenter.storage.content_storage)

                        #print("바뀐 후")
                        #for j in range(len(self.dataCenter.storage.content_storage)):
                        #    print(self.dataCenter.storage.content_storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1
                        #print("===========================DC Cache changed===========================")
        
        elif action == 3:
            self.action_3_cnt += 1