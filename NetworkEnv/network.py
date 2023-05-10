#import config as cf
#import node as nd
import random
import math
import random
import numpy as np
import NetworkEnv.caching as ch
import NetworkEnv.config as cf
import NetworkEnv.content as ct
import NetworkEnv.scenario as sc
import NetworkEnv.node as nd
import NetworkEnv.savingfunctions as sf

class Network(list):
    def __init__(self, Scenario, args):
        # implementing Scenario
        self.scenario = Scenario
        #self.sc = Scenario

        # set network variables
        self.set_network_variables()

        # caching algorithm configuration
        #self.cache_allocation_algorithm = args.allocation_algorithm
        #self.cache_replacement_algorithm = args.replacement_algorithm

        #print('cache_allocation_algorithm : ',self.cache_allocation_algorithm)
        #print('cache_replacement_algorithm : ',self.cache_replacement_algorithm)


        self.action_space = np.array([0,1,2,3])
        self.set_DQN_variables()

    def set_cache_hit_count(self):
        self.total_cache_hit_count = 0

    def get_cache_hit_count(self):
        return self.total_cache_hit_count

    def update_total_cache_hit_count(self):
        if cf.CORE_ID not in self.path:
            self.total_cache_hit_count += 1

    def set_total_hop(self):
        self.total_hop = 0

    def get_total_hop(self):
        return self.total_hop

    def update_hop(self, hop):
        self.total_hop += hop

    def set_total_latency(self):
        self.total_latency = 0

    def get_total_latency(self):
        return self.total_latency
    
    def update_total_latency(self, latency):
        self.total_latency += latency

    def set_content_diversity(self):
        self.content_diversity = 0

    def get_content_diversity(self):
        return self.content_diversity
    
    def update_content_diversity(self):
        ct_list = []
        for idx in range(cf.NB_NODES, cf.coreidx):
            for ct_idx in range(len(self.getBS(idx).storage.content_storage)):
                ct_list.append(self.getBS(idx).storage.content_storage[ct_idx].id)

        
        # 네트워크 내에 있는 모든 content의 갯수
        ct_cnt = len(ct_list)
        
        # identical 한 content 의 갯수 세기.
        ct_list = set(ct_list)
        ct_list = list(ct_list)
        
        identical_ct_cnt = len(ct_list)

        # content_diversity 구하기.
        self.content_diversity = identical_ct_cnt/ct_cnt
        
    def set_content_redundancy(self):
        self.cotent_redundancy = 0

    def get_content_redundancy(self):
        return self.cotent_redundancy

    def update_content_redundancy(self):
        ct_list = []
        for idx in range(cf.NB_NODES, cf.coreidx):
            for ct_idx in range(len(self.getBS(idx).storage.content_storage)):
                ct_list.append(self.getBS(idx).storage.content_storage[ct_idx].id)


        # 네트워크 내에 있는 모든 content의 갯수
        ct_cnt = len(ct_list)

        # identical 한 content 의 갯수 세기.
        ct_list = set(ct_list)
        ct_list = list(ct_list)
 
        identical_ct_cnt = len(ct_list)

        # content_redundancy 구하기.
        self.content_redundancy = 1 - (identical_ct_cnt/ct_cnt)
        print(self.content_redundancy)

    def print_results(self):
        print(f'round : {self.round+1}')
        print('avg_latency : {}'.format(self.get_total_latency()/cf.MAX_ROUNDS))
        print('avg_hop : {}'.format(self.get_total_hop()/cf.MAX_ROUNDS))
        print('Cache Hit Ratio : {}'.format(self.get_cache_hit_count()/cf.MAX_ROUNDS))

    def generate_days(self):
        total_day = 7*cf.TOTAL_PRIOD
        days = random.choices(range(total_day), k=cf.MAX_ROUNDS)
        days.sort()
        self.days = days

    def simulate(self):
        for round_nb in range(cf.MAX_ROUNDS):
            self.round= round_nb
            self.date = self.days[self.round]
            round_day = self.date % 7
            self.run_round(round_day)
            

    def cahcing_algorithm_run_round(self, _day):
        self.requested,self.path = self.request_and_get_path(_day)
        self.update_results()
        #! delete expiered date contents
        self.caching()

    def caching(self):
        ch.caching(self)

    #def delete_expiered_content(self):

    def update_results(self):
        self.update_total_cache_hit_count()
        self.update_hop(len(self.path) - 1)
        self.update_total_latency(self.get_latency(self.path))

    def search_parent_node(self,x,y,index):
        #type node:0, microbs:1 bs:2
        if index < cf.microStartidx:
            type = 0
        elif index < cf.bsStartidx:
            type = 1
        if type is 0:
            minRange = cf.AREA_LENGTH
            closestNode:nd.microBS
            closestID:int
            for i in self.microBSList:
                range =  math.sqrt(math.pow((x-i.pos_x),2) + math.pow((y-i.pos_y),2))
                if minRange>range:
                    closestNode=i
                    minRange=range
                    closestID=closestNode.id

        if type is 1:
            minRange = cf.AREA_LENGTH
            closestNode:nd.BS
            closestID:int
            for i in self.BSList:
                range =  math.sqrt(math.pow((x-i.pos_x), 2) + math.pow((y-i.pos_y),2))
                if minRange>range:
                    closestNode=i
                    minRange=range
                    closestID=closestNode.id
                    
        return closestID

    def hierarchical_request_and_get_path(self,_day):
        path=[]
        #시작 
        id = random.choice(range(0,cf.NB_NODES))
        time_delay = 0 
        #요청 content 선택
        requested_content = self.scenario.requestGenerate(_day)
        path.append(id)#노드
        
        micro_hop = self.search_parent_node(self.nodeList[id].pos_x,self.nodeList[id].pos_y,path[-1])
        path.append(micro_hop)#microBS
        if self.search(self.microBSList,micro_hop).storage.isstored(requested_content)==0:
            bs_hop = self.search_parent_node(self.search(self.microBSList,micro_hop).pos_x, self.search(self.microBSList,micro_hop).pos_y, path[-1])
            path.append(bs_hop)#BS
            if self.search(self.BSList, bs_hop).storage.isstored(requested_content)==0:
                path.append(cf.DATACENTER_ID)#center
                if self.dataCenter.storage.isstored(requested_content)==0:
                    path.append(cf.CORE_ID)
        
        return requested_content,path 
        
    def request_and_get_path(self,_day):
        path=[]
        #시작 
        id = random.choice(range(0,cf.NB_NODES))
        time_delay = 0 
        #요청 content 선택
        requested_content = self.scenario.requestGenerate(self.date,_day)
        path.append(id)#노드
        #바로 윗micro BS 탐색
        micro_hop = self.search_parent_node(self.nodeList[id].pos_x,self.nodeList[id].pos_y,path[-1])
        path.append(micro_hop)#microBS
        #요청한 컨텐츠 없을경우 parent BS 탐색
        if self.search(self.microBSList,micro_hop).storage.isstored(requested_content)==0:
            bs_hop = self.search_parent_node(self.search(self.microBSList,micro_hop).pos_x,self.search(self.microBSList,micro_hop).pos_y, path[-1])
            path.append(bs_hop)#BS
            #없으면 아래 micro BS 탐색
            next_microPath = 0
            if self.search(self.BSList, bs_hop).storage.isstored(requested_content)==0:
                #print(bs_hop,cf.bsStartidx)
                for i in self.mbsToBSLink[bs_hop-cf.bsStartidx]:
                    if self.search(self.microBSList,i).storage.isstored(requested_content)==1:
                        #print(self.search(self.microBSList,i).storage.content_storage)
                        next_microPath=i
                if next_microPath is not 0:
                    path.append(next_microPath)
                elif next_microPath is 0:
                    path.append(cf.DATACENTER_ID)#center
                    if self.dataCenter.storage.isstored(requested_content)==0:
                        path.append(cf.CORE_ID)
        #print("최종path:",path)
        #print("requested content",requested_content.__dict__)

        return requested_content,path

    def get_forward_transmission_time(self,index_i,index_j):
        # forward 에서는 ack delay time 을 계산
        # 마지막 부분은 3 hand shake를 구현

        # 아직 수식 구현에 있어서 변수 값 정해줘야함.
        
        i_x, i_y = self.checkBS_and_getPosition(index_i)
        j_x, j_y = self.checkBS_and_getPosition(index_j)

        # DATACENTER 와 CORE 인 경우 둘다 (0,0) 좌표
        # Latency 는 config Latency_Internet
        if (i_x == 0) & (i_y == 0) & (j_x == 0) & (j_y ==0):
            print('datacenter <-> core')
            return cf.LATENCY_INTERNET

        # uplink latency
        if index_i < index_j:
            print('uplink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.1, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.ACK_PACKET_SIZE/cf.ULthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.ACK_PACKET_SIZE/cf.ULthroughput
            
        # downlink latency
        else:
            print('downlink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.3, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.ACK_PACKET_SIZE/cf.DLthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.ACK_PACKET_SIZE/cf.DLthroughput

        return propagation_delay+transmission_delay+queuing_delay
    
    def get_backward_transmission_time(self,index_i,index_j):
        # index 를 보고 Node/MicroBS/BS/DataCenter/Core 인지 확인 후 transmission time 결정
        transmission_time = 0
        
        i_x, i_y = self.checkBS_and_getPosition(index_i)
        j_x, j_y = self.checkBS_and_getPosition(index_j)


        # DATACENTER 와 CORE 인 경우 둘다 (0,0) 좌표
        # Latency 는 config Latency_Internet
        if (i_x == 0) & (i_y == 0) & (j_x == 0) & (j_y ==0):
            print('datacenter <-> core')
            return cf.LATENCY_INTERNET

        # uplink latency
        if index_i < index_j:
            print('uplink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.1, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.PACKET_SIZE/cf.ULthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.PACKET_SIZE/cf.ULthroughput
            
        # downlink latency
        else:
            print('downlink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.3, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.PACKET_SIZE/cf.DLthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.PACKET_SIZE/cf.DLthroughput

        return propagation_delay+transmission_delay+queuing_delay

    def get_transmission_time(self,index_i,index_j):
        # index 를 보고 Node/MicroBS/BS/DataCenter/Core 인지 확인 후 transmission time 결정
        transmission_time = 0
        
        i_x, i_y = self.checkBS_and_getPosition(index_i)
        j_x, j_y = self.checkBS_and_getPosition(index_j)


        # DATACENTER 와 CORE 인 경우 둘다 (0,0) 좌표
        # Latency 는 config Latency_Internet
        if (i_x == 0) & (i_y == 0) & (j_x == 0) & (j_y ==0):
            #print('datacenter <-> core')
            return cf.LATENCY_INTERNET

        # uplink latency
        if index_i < index_j:
            #print('uplink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.1, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.PACKET_SIZE/cf.ULthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.PACKET_SIZE/cf.ULthroughput
            
        # downlink latency
        else:
            #print('downlink : {'+ str(index_i) + '} -> {' + str(index_j) + '}')
            traffic_intensity = 1-abs(np.random.normal(0, 0.3, 1))
            range = math.sqrt(math.pow(i_x - j_x, 2) + math.pow(i_y - j_y, 2))
            propagation_delay = range/ cf.LIGHT_SPEAD
            transmission_delay = cf.PACKET_SIZE/cf.DLthroughput
            queuing_delay = traffic_intensity*(1-traffic_intensity)*cf.PACKET_SIZE/cf.DLthroughput

        #print(propagation_delay+transmission_delay+queuing_delay)

        return propagation_delay+transmission_delay+queuing_delay

    def get_latency(self, path):

        # index NODE 부터 CORE 까지 0부터 시작
        # 따라서 n번째 index 가 n+1번째 index 보다 작으면 uplink latency
        # n번째 index 가 n+1번째 index 보다 크면 downlink latency
        
        latency = 0
        #print('===forward===')
        #print('path : {}'.format(path))
        # forward
        for n in range(len(path)-1):
            latency += self.get_transmission_time(path[n],path[n+1])
        
        # backward
        path.reverse()
        #print('===backward===')
        #print('path : {}'.format(path))
        for n in range(len(path)-1):
            latency += self.get_transmission_time(path[n],path[n+1])

        # 다시 원상복구
        path.reverse()

        #print('total latency : {}'.format(latency))

        return latency


# 내가 만든 함수들 
# 목록 : reset, request, get_simple_path, get_c_nodeList

    def reset(self):
        self.__init__()

    def get_simple_path(self, nodeId):

        path=[]
        #시작 
        id = nodeId
        path.append(id)#노드
        # 노드 x,y 좌표를 통해 [node - micro - BS - Data center - Core Internet]
        micro_hop = self.search_parent_node(self.nodeList[id].pos_x,self.nodeList[id].pos_y,path[-1])
        path.append(micro_hop)#microBS
        bs_hop = self.search_parent_node(self.search(self.microBSList,micro_hop).pos_x, self.search(self.microBSList,micro_hop).pos_y, path[-1])
        path.append(bs_hop)# Base Station
        path.append(cf.dcidx)# Data Center
        path.append(cf.coreidx)# Core Internet
        return path

    def get_c_nodeList(self):

        # TODO : Core Internet --> 모든 노드
        # TODO : Data Center --> 모든 노드
        # 각각 따로 for 문이 돌아갈 필요 X
        for id in range(cf.NB_NODES):
            self.CoreNodeList.append(id)
            self.DataCenterNodeList.append(id)

        # TODO : 먼저 모든 노드들의 path를 구한뒤 배열로 각각 따로 저장하자
        # TODO : Micro Base Station --> Node들을 저장
        # TODO : Base Station --> 연결 되어있는 Micro Base Station 저장

        nodePathList = []
        tmpPath = []
        for id in range(cf.NB_NODES):
            tmpPath = self.get_simple_path(id)
            #print(tmpPath)
            nodePathList.append(tmpPath)
            tmpPath = []
        
        #print(nodePathList)

        # MicroBS_Id 는 cf.NB_NODES ~ cf.NUM_microBS[0]*cf.NUM_microBS[1] - 1 범위에 있다.
        for MicroBS_Id in range(cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1]):

            tmpMicroNodeList = []
            if MicroBS_Id < cf.NB_NODES:
                tmpMicroNodeList.append(-1)

            else:
                for i in range(cf.NB_NODES):
                    # nodePathList = [[0, 64, 7, 0, 0], ... , [300, 5, 2, 0, 0]]
                    # MicroNodePathList 에는 MicroBS 의 id 가 index 
                    # 해당 index 에 node id 들이 append 됌
                    
                    if MicroBS_Id == nodePathList[i][1]:
                        #print("node의 id : " + str(nodePathList[i][0]) + " 추가")
                        tmpMicroNodeList.append(nodePathList[i][0])

                if len(tmpMicroNodeList) == 0:
                    tmpMicroNodeList.append(-1)
            #print("MicroBSID 에 포함되는 NodeList : " + str(tmpMicroNodeList))
            self.MicroBSNodeList.append(tmpMicroNodeList)
        
        #print('self.MicroBSNodeList : {}'.format(self.MicroBSNodeList))
        # BS_Id 는 cf.NUM_microBS[0]*cf.NUM_microBS[1] ~ cf.NUM_BS[0]*cf.NUM_BS[1] - 1 범위에 있다.
        for BS_Id in range(cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1] + cf.NUM_BS[0]*cf.NUM_BS[1]):
            tmpBSNodeList = []
            if BS_Id < (cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1]):
                tmpBSNodeList.append(-1)

            else:
                for i in range(cf.NB_NODES):
                    # BSNodePathList 에는 BS 의 id 가 index 
                    # 해당 index 에 MicroBS id 들이 append 됌
                    if BS_Id == nodePathList[i][2]:

                        if nodePathList[i][1] not in tmpBSNodeList:
                            tmpBSNodeList.append(nodePathList[i][1])

                if len(tmpMicroNodeList) == 0:
                    tmpBSNodeList.append(-1)
            #print(tmpBSNodeList)
            self.BSNodeList.append(tmpBSNodeList)
        #print('self.BSNodeList : {}'.format(self.BSNodeList))

    def search(self, _list:list, _id):
        for index, element in enumerate(_list):
             if element.id == _id:
                return element

    def checkBS(self, _id):
        # NODE
        if _id < cf.NB_NODES:
            return 'NODE'
        # MicroBS
        elif _id < cf.NB_NODES + cf.NUM_microBS[0] * cf.NUM_microBS[1]:
            return 'MicroBS'
        # BS
        elif _id < cf.NB_NODES + cf.NUM_microBS[0] * cf.NUM_microBS[1] + cf.NUM_BS[0]*cf.NUM_BS[1]:
            return 'BS'
        # DataCenter
        elif _id == cf.DATACENTER_ID:
            return 'DataCenter'
        # Core
        else:
            return 'CORE'
    
    def checkBS_and_getPosition(self, _id):
        # NODE
        if _id < cf.NB_NODES:
            #print('NODE')
            element = self.search(self.nodeList, _id)
            return element.pos_x, element.pos_y
        # MicroBS
        elif _id < cf.NB_NODES + cf.NUM_microBS[0] * cf.NUM_microBS[1]:
            #print('MicroBS')
            element = self.search(self.microBSList, _id)
            return element.pos_x, element.pos_y
        # BS
        elif _id < cf.NB_NODES + cf.NUM_microBS[0] * cf.NUM_microBS[1] + cf.NUM_BS[0]*cf.NUM_BS[1]:
            #print('BS')
            element = self.search(self.BSList, _id)
            return element.pos_x, element.pos_y
        # DataCenter
        elif _id == cf.DATACENTER_ID:
            #print('DataCenter')
            return self.dataCenter.pos_x, self.dataCenter.pos_y
        # Core
        else:
            #print('CORE')
            return 0, 0

    def getBS(self, BS_id):
        BS_type = self.checkBS(BS_id)
        if BS_type == 'MicroBS':
            return self.search(self.microBSList, BS_id)
        elif BS_type == 'BS':
            return self.search(self.BSList, BS_id)
        elif BS_type == 'DataCenter':
            return self.dataCenter
        else:
            print('It\'s NODE or CORE.')

    def mbsLink(self):

        temp_BS=[]
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            temp_BS.append([])
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            BS_index = self.search_parent_node(self.microBSList[i].pos_x,self.microBSList[i].pos_y,i+cf.microStartidx)-cf.bsStartidx
            temp_BS[BS_index].append(i+cf.microStartidx)
            
        self.mbsToBSLink=temp_BS
        #print(self.mbsToBSLink)

# Functions for Deep Reinforcement Learning

    def set_network_variables(self):
        self.nodeList = nd.generateNode()
        self.microBSList = nd.generateMicroBS()
        self.BSList = nd.generateBS()
        self.dataCenter = nd.dataCenter(0,0)
        
        self.mbsToBSLink=[]

        self.CoreNodeList = []
        self.DataCenterNodeList = []
        self.BSNodeList = []
        self.MicroBSNodeList = []
        self.days = []

        self.get_c_nodeList()
        self.generate_days()
        self.requested_content:ct.Content
        self.mbsLink()

        # For network caching algorithm comparsion
        self.set_cache_hit_count()
        self.set_total_hop()
        self.set_total_latency()

    def set_DQN_variables(self):
        # for action, state_dim declaration
        self.action_space = np.array([0,1,2,3])
        self.state_dim = 105

        # for cases 
        self.case_A=0
        self.case_B=0
        self.case_C=0
        self.case_D=0
        self.case_E=0
        self.case_F=0
        self.case_G=0
        self.case_H=0
        self.case_I=0
        self.case_J=0
        self.case_K=0
        self.case_L=0
        self.case_M=0
        self.case_N=0
        self.case_O=0
        self.case_P=0
        self.case_Q=0
        self.case_R=0
        
        # path는 [node, Micro BS, BS, Data center, Core Internet]
        self.path = []

        # request dictionary
        self.requestDictionary = self.set_requestDictionary()
        self.contentLabelDictionary = self.set_contentLabelDictionary()
        self.actionDictionary = self.set_actionDictionary()
        self.roundRequestDictionary = self.set_roundRequestDictionary()

        self.round_nb = 0
        self.round_day = 0
        self.state:np.array
        

        # save the results
        #self.save_epi_reward = []
        #self.save_epi_cache_hit_rate = []
        #self.save_epi_redundancy = []
        #self.save_epi_avg_hop = []
        #self.save_epi_existing_content = []
        #self.save_epi_denominator = []

        # reward parameter
        self.a = cf.a
        self.b = cf.b
        self.c = cf.c
        self.d = cf.d
        self.e = cf.e

        self.d_core = 0
        self.d_cache = 0
        self.R_cache = 0
        self.H_arg = 0
        self.c_node = 0
        self.stored_type = 0
        self.stored_nodeID = 0
        self.alpha_redundancy = 0
        self.beta_redundancy = 0
        self.vacancy = 0

        # Done 조건 action이 7000번 일어나면 끝
        self.NB_ACTION = cf.NB_ACTION
        self.stop = cf.MAX_ROUNDS - 2
        self.action_cnt = 0
        self.step_cnt = 0
        
        # saving function
        self.sf = sf.savingfunctions()

        # cache hit count ==> network.py에 넣어야할지도 모름
        self.total_cache_hit_cnt = 0
        self.hop_cnt = 0

        # cache 교체 Flag [0,1]
        # 0 : 교체 안됌
        # 1 : 교체 됌
        self.cache_changed = 0
        self.cache_changed_cnt = 0
        self.action_3_cnt = 0
        self.date = 0
        self.lastday = 0

        self.gamma_episode_reward = 0
        
        #state 바꾸는 중
        self.current_requested_content:ct.Content 
        self.current_path = []
        self.current_full_path = []
        self.next_requested_content:ct.Content 
        self.next_path = []

        # action count 변수
        self.random_action_cnt_list = self.set_action_list()
        self.qs_action_cnt_list = self.set_action_list()

    def set_state(self):
       
        # state = [
        #   requested_content, request count of requested_content,
        #   MicroBS: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        #   .
        #   .
        #   BS: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        #   .
        #   .
        #   DataCenter: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        # ]
        state = []
       
        #title = self.current_requested_content.get_title()
        Rct_id = self.current_requested_content.id
        #! round_nb 수정
        date = self.days[self.round_nb]
        day = date%7

        state.append(Rct_id)

        for period in range(4):
            state.append(self.roundRequestDictionary[Rct_id][(day+period)%7])

        # MicroBS [cache available, Labels of contents in storage]
        content_cnt = len(self.search(self.microBSList, self.current_full_path[1]).storage.content_storage)
        for j in range(content_cnt):

            ct_id = self.search(self.microBSList, self.current_full_path[1]).storage.content_storage[j].id
            #content_label = self.contentLabelDictionary[ct_id]
            state.append(ct_id)

            for period in range(4):
                content_roundrequest_count = self.roundRequestDictionary[ct_id][(day+period)%7]
                state.append(content_roundrequest_count)
   
        # storage 가 꽉 차있지 않을 때, 아무것도 없다는 label = 0 을 넣어줌
        while content_cnt != cf.microBS_SIZE/cf.CONTENT_SIZE:
            content_cnt = content_cnt + 1
            # content label
            state.append(0)
            # content round request count
            for i in range(4):
                state.append(0)

        # BS
        content_cnt = len(self.search(self.BSList,self.current_full_path[2]).storage.content_storage)

        for j in range(content_cnt):
            ct_id = self.search(self.BSList, self.current_full_path[2]).storage.content_storage[j].id
            #content_label = self.contentLabelDictionary[title]
            state.append(ct_id)
            for period in range(4):
                content_roundrequest_count = self.roundRequestDictionary[ct_id][(day+period)%7]
                state.append(content_roundrequest_count)

        # storage 가 꽉 차있지 않을 때, 아무것도 없다는 label = 0 을 넣어줌
        while content_cnt != cf.BS_SIZE/cf.CONTENT_SIZE:          
            content_cnt = content_cnt + 1
            # content label
            state.append(0)
            # content round request count
            for day in range(4):
                state.append(0)

        # DataCenter
        content_cnt = len(self.dataCenter.storage.content_storage)
        for i in range(content_cnt):
            ct_id = self.dataCenter.storage.content_storage[i].id
            #content_label = self.contentLabelDictionary[title]
            state.append(ct_id)

            for period in range(4):
                content_roundrequest_count = self.roundRequestDictionary[ct_id][(day+period)%7]
                state.append(content_roundrequest_count)
  
        # storage 가 꽉 차있지 않을 때, 아무것도 없다는 label = 0 을 넣어줌
        while content_cnt != cf.CENTER_SIZE/cf.CONTENT_SIZE:          
            content_cnt = content_cnt + 1
            # content label
            state.append(0)
            # content request count
            for day in range(4):
                state.append(0)

        state = np.array(state)
        return state

    def check_done(self):

        if self.round_nb == cf.MAX_ROUNDS - 1:
            return 1
        else:
            return 0

    def run_round(self):
        #! self.env.run_round()에서 수행해야할 것.
        #! round_day 올려주는 것
        #! network state 반환
        #! requested content, path 반환
        #! 홉 계산
        #! delete period 지난 것에 대한 수행

        done = self.check_done()
        self.date = self.days[self.round_nb]
        round_day = self.days[self.round_nb] % 7
        requested_content, self.path = self.request_and_get_path(round_day)
        
        self.current_requested_content = requested_content
        self.current_full_path = self.get_simple_path(self.path[0])
        #!
        #title = self.current_requested_content.get_title()
        requested_content_id = self.current_requested_content.id
        
        #self.update_requestDictionary(title)
        self.updateRoundRequestDictionary(requested_content_id, round_day)
        self.hop_cnt += len(self.path) - 1

        if self.date != self.lastdate:
            #print('delete out of day content')
            #print('current date is ', self.date)
            #print('BBBBBBBBBBBBBEFOREEEEEEEEEEEEE')
            #self.showAllStorage()
            self.deleteOutofdateContents()
            #print('AAAAAAAAAAAAAFTERRRRRRRRRRRRRR')
            #self.showAllStorage()

        self.lastdate = self.date

        state = self.set_state()

        #print('self.round_nb : {}'.format(self.round_nb))
        #print('self.date : {}'.format(self.date))
        #print('state : {}'.format(state))
        #print(requested_content.__dict__)
        #print('path : {}'.format(self.path))
        if len(self.path) > 4:
            CACHE_HIT_FLAG = 0
            #print("Cache didn't Hit!!!")
            return CACHE_HIT_FLAG, state, requested_content, self.path, done
        else:
            CACHE_HIT_FLAG = 1
            self.total_cache_hit_cnt += 1
            #print("Cache Hitted!!!")
            return CACHE_HIT_FLAG, None, None, None, done

    def step(self, action, path, requested_content):
        self.step_cnt = self.step_cnt + 1
        # 이제 여기서 요청 시작 {노드, 요청한 컨텐츠}
        self.action_cnt = self.action_cnt + 1
        #ispathempty
        self.ispathempty_result = self.ispathempty()
        #ispathchangeable
        self.ispathchangeable_result = self.ispathchangeable()
        self.act(path, requested_content, action)

        next_state = self.set_state()
        self.round_nb += 1
        reward = self.get_reward(action, path, requested_content)

        return next_state, reward 
    
    def update(self):
        self.updatequeue()
        #print('updated state: {}'.format(self.set_state()))
        self.round_nb += 1
    
    def reset_parameters(self):
        #! network init 다시 한번 해야함.
        #self.__init__()
        self.round_nb, self.total_cache_hit_cnt, self.action_cnt, self.step_cnt, self.hop_cnt = 0,0,0,0,0
        self.date = 0
        self.lastdate = 0

        self.gamma_episode_reward = 0

        self.cache_changed_cnt = 0
        self.action_3_cnt = 0

        self.requestDictionary = self.set_requestDictionary()
        self.actionDictionary = self.set_actionDictionary()
        self.roundRequestDictionary = self.set_roundRequestDictionary()

        self.random_action_cnt_list = self.set_action_list()
        self.qs_action_cnt_list = self.set_action_list()

        self.case_A=0
        self.case_B=0
        self.case_C=0
        self.case_D=0
        self.case_E=0
        self.case_F=0
        self.case_G=0
        self.case_H=0
        self.case_I=0
        self.case_J=0
        self.case_K=0
        self.case_L=0
        self.case_M=0
        self.case_N=0
        self.case_O=0
        self.case_P=0
        self.case_Q=0
        self.case_R=0

#! HETEROGENOUS CONETENT SIZE 에 대한 act 를 재정의 하여야함.
#! set_state 에 대한 정의를 먼저하고 나서 진행하여야할듯함.
    def heterogenous_act(self, path, requested_content, action):

        # self.round_day 기준으로 3일 뒤까지 requested 된 횟수를 가지고 비교 후 caching 할지 말지 정함
        
        # self.roundRequestDictionary : [일, 월, 화, 수, 목, 금, 토] : 각 날짜 별 request 횟수 저장함
        # self.roundRequestDictionary = { "title_1" = [0,2,0,14,20,3,4], ... , "title_N" = [30,5,0,0,0,3,8]}
        # round_day 가 0이면 0,1,2 즉 일,월,화 의 requested 된 횟수의 합을 가지고 비교함
        date = self.days[self.round_nb]
        round_day =  date % 7
        print("act 들어옴")
        requested_content:ct.Content = requested_content
        
        requested_content_id = requested_content.id
        tmp_round_request = 0
        roundRequested_cnt_list = []
        path = path

        requested_content_round_cnt = self.roundRequestDictionary[requested_content_id][round_day] + self.roundRequestDictionary[requested_content_id][(round_day+1)%7] + self.roundRequestDictionary[requested_content_id][(round_day+2)%7] + self.roundRequestDictionary[requested_content_id][(round_day+3)%7]
        #print("\n\n")
        #print("requested content : {}".format(requested_content_id))
        #print("round_day : {}".format(round_day))
        #print(requested_content_id + " : " + str(self.roundRequestDictionary[requested_content_id]))
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
                    while self.search(self.microBSList, path[1]).storage.abletostore(requested_content):

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


                            print("del_content : {}".format(del_content.__dict__))
                            print("바뀌기 전")
                            
                            for j in range(len(self.search(self.microBSList, path[1]).storage.content_storage)):
                                print(self.search(self.microBSList, path[1]).storage.content_storage[j].__dict__)

                            self.search(self.microBSList, path[1]).storage.delContent(del_content)
                            self.search(self.microBSList, path[1]).storage.addContent(requested_content)
                            requested_content.update_validity(date)

                            print("바뀐 후")
                            for j in range(len(self.search(self.microBSList, path[1]).storage.content_storage)):
                                print(self.search(self.microBSList, path[1]).storage.content_storage[j].__dict__)

                            self.cache_changed = 1
                            self.cache_changed_cnt +=1

                        print("===========================MBS Cache changed===========================")


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
                    while self.search(self.BSList, path[1]).storage.abletostore(requested_content):
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
                    while self.dataCenter.storage.abletostore(requested_content):
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

    def act(self, path, requested_content, action):

        # self.round_day 기준으로 3일 뒤까지 requested 된 횟수를 가지고 비교 후 caching 할지 말지 정함
        
        # self.roundRequestDictionary : [일, 월, 화, 수, 목, 금, 토] : 각 날짜 별 request 횟수 저장함
        # self.roundRequestDictionary = { "title_1" = [0,2,0,14,20,3,4], ... , "title_N" = [30,5,0,0,0,3,8]}
        # round_day 가 0이면 0,1,2 즉 일,월,화 의 requested 된 횟수의 합을 가지고 비교함
        date = self.days[self.round_nb]
        round_day =  date % 7

        requested_content:ct.Content = requested_content
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


    def requested_content_and_get_path(self, nodeID, requested_content):
        path=[]
        #시작 
        id = nodeID
        time_delay = 0 
        #요청 content 선택
        requested_content = requested_content
        path.append(id)#노드
        
        # 노드 x,y 좌표를 통해 [micro - BS - Data center - Core Internet]
        micro_hop = self.search_parent_node(self.nodeList[id].pos_x,self.nodeList[id].pos_y,0)
        path.append(micro_hop)#microBS

        if self.search(self.microBSList, micro_hop).storage.isstored(requested_content)==0:
            
            bs_hop = self.search_parent_node(self.search(self.microBSList, micro_hop).pos_x,self.search(self.microBSList, micro_hop).pos_y, micro_hop)
            #print(bs_hop)
            path.append(bs_hop)#BS
            if self.search(self.BSList,bs_hop).storage.isstored(requested_content)==0:
                path.append(cf.DATACENTER_ID)#center
                if self.dataCenter.storage.isstored(requested_content)==0:
                    path.append(cf.CORE_ID)
        return path

    def get_reward(self, action, path, requested_content):
        """
        Return the reward.
        The reward is:
        
            Reward = a*(d_core - d_cache) - b*(#ofnode - coverage_node)

            a,b = 임의로 정해주자 실험적으로 구하자
            d_core  : 네트워크 코어에서 해당 컨텐츠를 전송 받을 경우에 예상되는 지연 시간.
            d_cache : 가장 가까운 레벨의 캐시 서버에서 해당 컨텐츠를 받아올 때 걸리는 실제 소요 시간
            cf.NB_NODES : 노드의 갯수
            c_node : agent 저장할 때 contents가 있는 station이 포괄하는 device의 갯수
        """
        self.set_reward_parameter(path, requested_content=requested_content)
        #reward = 0
        if action == 3:
            if self.ispathempty_result[0]+self.ispathempty_result[1]+self.ispathempty_result[2]>0:
                reward = cf.N_REWARD #case A
                self.case_A= self.case_A+1
                #print("case_A")
            #빈공간이 없을때
            else:
                #컨텐츠 교체가 가능할때
                if self.ispathchangeable_result[0]+self.ispathchangeable_result[1]+self.ispathchangeable_result[2]>0:
                    reward = cf.N_REWARD #case B
                    self.case_B= self.case_B+1
                    #print("case_B")
                #컨텐츠 교체가 불가능할때
                else:
                    reward = cf.P_REWARD #case C
                    self.case_C= self.case_C+1
                    #print("action 3", cf.P_REWARD)
                    #print("case_C")

        # action : 0,1,2
        if action == 0:
            if self.ispathempty_result[0] == 1:
                reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                self.case_D= self.case_D+1
            if self.ispathempty_result[0] == 0:
                if self.ispathempty_result[1]+self.ispathempty_result[2]>0: 
                    if self.ispathchangeable_result[0]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                        self.case_E= self.case_E+1
                    if self.ispathchangeable_result[0]==0:
                        reward = cf.N_REWARD - self.e*self.vacancy
                        self.case_F= self.case_F+1
                if self.ispathempty_result[1]+self.ispathempty_result[2]==0:
                    if self.ispathchangeable_result[0]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) 
                        #print("action 0" ,self.a*(self.d_core - self.d_cache), self.b*math.log2(self.c_node))
                        self.case_G= self.case_G+1
                        
                    if self.ispathchangeable_result[0]==0:
                        reward = cf.N_REWARD - self.e*self.vacancy
                        self.case_H= self.case_H+1
        if action == 1:
            if self.ispathempty_result[1] == 1:
                reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                self.case_I= self.case_I+1
            if self.ispathempty_result[1] == 0:
                if self.ispathempty_result[0]+self.ispathempty_result[2]>0: 
                    if self.ispathchangeable_result[1]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                        self.case_J= self.case_J+1
                    if self.ispathchangeable_result[1]==0:
                        reward = cf.N_REWARD - self.e*self.vacancy
                        self.case_K= self.case_K+1
                if self.ispathempty_result[0]+self.ispathempty_result[2]==0:
                    if self.ispathchangeable_result[1]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                        #print("action 1", self.a*(self.d_core - self.d_cache), self.b*math.log2(self.c_node))
                        self.case_L= self.case_L+1
                        
                    if self.ispathchangeable_result[1]==0:
                        reward = cf.N_REWARD - self.e*self.vacancy
                        self.case_M= self.case_M+1
        if action == 2:
            if self.ispathempty_result[2] == 1:
                reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                self.case_N= self.case_N+1
            if self.ispathempty_result[2] == 0:
                if self.ispathempty_result[0]+self.ispathempty_result[1]>0: 
                    if self.ispathchangeable_result[2]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node)
                        self.case_O= self.case_O+1
                    if self.ispathchangeable_result[2]==0:
                        reward = cf.N_REWARD 
                        self.case_P= self.case_P+1
                if self.ispathempty_result[0]+self.ispathempty_result[1]==0:
                    if self.ispathchangeable_result[2]==1:
                        reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) 
                        #print("action 2",self.a*(self.d_core - self.d_cache), self.b*math.log2(self.c_node))
                        self.case_Q= self.case_Q+1
                        
                    if self.ispathchangeable_result[2]==0:
                        reward = cf.N_REWARD - self.e*self.vacancy        
                        self.case_R= self.case_R+1
                    #print("case_I")

        # cache_changed 초기화
        self.cache_changed = 0
        self.ispathchangeable_result =[]
        self.ispathempty_result = []
        #print('reward : ', reward)
        reward = float(reward)
        return reward

    def set_reward_parameter(self, path, requested_content):

        # d_core  : 네트워크 코어에서 해당 컨텐츠를 전송 받을 경우에 예상되는 지연 시간.
        #          

        # d_cache : 가장 가까운 레벨의 캐시 서버에서 해당 컨텐츠를 받아올 때 걸리는 실제 소요 시간
                   
        # R_cache : 네트워크에 존재하는 동일한 캐시의 수
        #           for 문으로 돌려야겠다

        # c_node   : 캐싱된 파일이 커버하는 노드의 수 (coverage node)
        #           
        nodeID = path[0]
        self.d_core = self.get_d_core(nodeID, requested_content)
        self.d_cache = self.get_d_cache(nodeID, requested_content)
        self.c_node = self.get_c_node()
        #self.alpha_redundancy, self.beta_redundancy = self.set_content_redundancy(requested_content)
        self.vacancy = self.cal_vacancy()
        self.mbsLink()

    def get_d_core(self,nodeID, requested_content):
        # 코어 인터넷까지 가서 가져오는 경우를 봐야함
        # path 뒤에 추가해서 구하자
        path = []
        path = self.requested_content_and_get_path(nodeID, requested_content)

        # [4,68] 일 경우 ---> [4,68, search_parent_node(microBS.x, microBS.y):BS, search_parent_node(BS.x, BS.y):Datacenter, search_parent_node(Datacenter.x, Datacenter.y):Core Internet]
        # path 다 채워질 떄까지 돌리자
        while len(path) != 5:

            # Micro에 캐싱되어 있는 경우, BS 추가
            if len(path) == 2:
                id = path[-1]
                closestID = self.search_parent_node(self.search(self.microBSList, id).pos_x,self.search(self.microBSList, id).pos_y,path[-1])
                path.append(closestID)

            # BS에 캐싱 되어 있는 경우, Data Center 추가
            elif len(path) ==  3:
                path.append(cf.DATACENTER_ID)

            # 데이터 센터에 캐싱이 되어 있는 경우, Core Internet 추가
            elif len(path) == 4:
                path.append(cf.CORE_ID)

        d_core = self.get_latency(path) * 1000
        
        return d_core

    def get_d_cache(self, nodeID, requested_content):
        # TODO : 가장 가까운 레벨의 캐시 서버에서 해당 컨텐츠를 받아올 때 걸리는 실제 소요 시간
        path = []
        path = self.requested_content_and_get_path(nodeID, requested_content)
        d_cache = self.get_latency(path) * 1000

        return d_cache

    def get_c_node(self):
        # TODO : agent 저장할 때 contents가 있는 station이 포괄하는 device의 갯수
        c_node = 0
        tmpcnt = 0

        # MicroBS
        if self.stored_type == 0:
            c_node = len(self.MicroBSNodeList[self.stored_nodeID])
        
        # BS
        elif self.stored_type == 1:
            for i in self.BSNodeList[self.stored_nodeID]:
                tmpcnt += len(self.MicroBSNodeList[i])
            c_node = tmpcnt

        # DataCenter
        elif self.stored_type == 2:
            c_node = cf.NB_NODES

        return c_node

    def set_roundRequestDictionary(self):
        title_list = self.scenario.titleList
        roundRequestdict = {}

        for i in range(len(title_list)):
            # [일 월 화 수 목 금 토]
            # request 수 보고 plot 할 거임
            roundRequestdict[title_list[i]] = [0,0,0,0,0,0,0]
        return roundRequestdict

    def updateRoundRequestDictionary(self, title, round_day):
        #print(title + ' , ' + str(round_day))
        self.roundRequestDictionary[title][round_day] += 1
        #print(self.roundRequestDictionary)

    def tmp_set_contentLabelDictionary(self):
        title_list = self.scenario.titleList
        dict = {}
        
        for i in range(len(title_list)):
            dict[title_list[i]] = i+1
        
        #print(dict)
        return dict
#!
    def set_contentLabelDictionary(self):
        title_list = self.scenario.titleList
        dict = {}
        
        for i in title_list:
            dict[i] = i+1
        
        #print(dict)
        return dict

    def set_requestDictionary(self):
        title_list = self.scenario.titleList
        #print(title_list)
        contentdict = {}

        for i in range(len(title_list)):
            contentdict[title_list[i]] = 0

        return contentdict

    def update_requestDictionary(self, title):
        self.requestDictionary[title] += 1

    def set_actionDictionary(self):
        title_list = self.scenario.titleList
        #print(title_list)
        actiondict = {}

        for i in range(len(title_list)):
            actiondict[title_list[i]] = []

        return actiondict

    def append_actionDictionary(self, title, action):
        self.actionDictionary[title].append(action)

    def cal_vacancy(self):

        vacancy = 0 
        #print("=========================MicroBS=========================")
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            #print(self.microBSList[i].storage.__dict__)
            #print("vacancy : {}".format(self.microBSList[i].storage.capacity - self.microBSList[i].storage.stored))
            vacancy += self.microBSList[i].storage.capacity - self.microBSList[i].storage.stored

        #print("=========================BS=========================")
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            #print(self.BSList[i].storage.__dict__)
            #print("vacancy : {}".format(self.BSList[i].storage.capacity - self.BSList[i].storage.stored))
            vacancy += self.BSList[i].storage.capacity - self.BSList[i].storage.stored

        #print("=========================Datacenter=========================")
        #print(self.dataCenter.storage.__dict__)
        #print("vacancy : {}".format(self.dataCenter.storage.capacity - self.dataCenter.storage.stored))
        vacancy += self.dataCenter.storage.capacity - self.dataCenter.storage.stored

        return vacancy     

    def showAllStorage(self):
        print("===============SHOW ALL STORAGE===============")
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            print("{}번째 Micro BS Storage".format(i))

            for j in range(len(self.microBSList[i].storage.content_storage)):
                content = self.microBSList[i].storage.content_storage[j]
                print(content.__dict__)
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            print("{}번째 BS Storage".format(i))
            for j in range(len(self.BSList[i].storage.content_storage)):
                content = self.BSList[i].storage.content_storage[j]
                print(content.__dict__)
        # DataCenter
        print("DataCenter Storage")
        for j in range(len(self.dataCenter.storage.content_storage)):
            content = self.dataCenter.storage.content_storage[j]
            print(content.__dict__)

        print("===============FINISH SHOW ALL STORAGE===============")

    #! requested date 를 따로 어떻게 지정할지 생각해야함.
    #! 내가 봤을 땐, content의 dictionary 안에 requested date 하나 만들어서 업데이트하고 해당 키값 통해서 deleteOutofdateContents() 실행해야할듯.             
    
    def deleteOutofdateContents(self):
        
        delcontentlist = []
        #MicroBS
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            for j in range(len(self.microBSList[i].storage.content_storage)):
                content = self.microBSList[i].storage.content_storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.microBSList[i].storage.lastdatelist[j]))
                content_expired_date = content.get_expired_date()

                if content_expired_date <= self.date:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.microBSList[i].storage.delContent(delcontent)

            delcontentlist = []
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            
            for j in range(len(self.BSList[i].storage.content_storage)):
                content = self.BSList[i].storage.content_storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.BSList[i].storage.lastdatelist[j]))
                content_expired_date = content.get_expired_date()
                
                if content_expired_date <= self.date:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.BSList[i].storage.delContent(delcontent)
            delcontentlist = []

        # DataCenter
        for j in range(len(self.dataCenter.storage.content_storage)):
            content = self.dataCenter.storage.content_storage[j]
            #print(str(content.__dict__), " ==> lastdate : ", str(self.dataCenter.storage.lastdatelist[j]))
            content_expired_date = content.get_expired_date()
            if content_expired_date <= self.date:
                delcontentlist.append(content)
                
        for delcontent in delcontentlist:
            self.dataCenter.storage.delContent(delcontent)
        delcontentlist = []
    
    
    
    def tmp_deleteOutofdateContents(self):
        
        limitdate = self.date - cf.DELETE_PERIOD
        delcontentlist = []
        #MicroBS
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            
            for j in range(len(self.microBSList[i].storage.content_storage)):
                content = self.microBSList[i].storage.content_storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.microBSList[i].storage.lastdatelist[j]))
                contentlastdate = self.microBSList[i].storage.lastdatelist[j]
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.microBSList[i].storage.delContent(delcontent)

            delcontentlist = []
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            
            for j in range(len(self.BSList[i].storage.content_storage)):
                content = self.BSList[i].storage.content_storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.BSList[i].storage.lastdatelist[j]))
                contentlastdate = self.BSList[i].storage.lastdatelist[j]
                
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.BSList[i].storage.delContent(delcontent)
            delcontentlist = []

        # DataCenter
        for j in range(len(self.dataCenter.storage.content_storage)):
            content = self.dataCenter.storage.content_storage[j]
            #print(str(content.__dict__), " ==> lastdate : ", str(self.dataCenter.storage.lastdatelist[j]))
            contentlastdate = self.dataCenter.storage.lastdatelist[j]
            if contentlastdate < limitdate:
                delcontentlist.append(content)
                
        for delcontent in delcontentlist:
            self.dataCenter.storage.delContent(delcontent)
        delcontentlist = []

    def set_action_list(self):
        
        action_list = [0,0,0,0]

        return action_list
    
    def ispathempty(self): #1이면 비어있어서 저장 가능
        isfull=[]
        isfull.append(self.search(self.microBSList,self.current_full_path[1]).storage.abletostore(self.current_requested_content))
        isfull.append(self.search(self.BSList, self.current_full_path[2]).storage.abletostore(self.current_requested_content))
        isfull.append(self.dataCenter.storage.abletostore(self.current_requested_content))
        return isfull

    def ispathchangeable(self): #1이면 변경 가능
        #!
        Rct_id = self.current_requested_content.id
        round_day = self.date % 7
        content_roundrequest_count = self.roundRequestDictionary[Rct_id][round_day] + self.roundRequestDictionary[Rct_id][(round_day+1)%7] + self.roundRequestDictionary[Rct_id][(round_day+2)%7] + self.roundRequestDictionary[Rct_id][(round_day+3)%7]
        result = []

        temp_result = 0
        if self.search(self.microBSList, self.current_full_path[1]).storage.isstored(self.current_requested_content) ==1:
            temp_result = 0    
        else:    
            for i in self.search(self.microBSList, self.current_full_path[1]).storage.content_storage:
                storage_content_roundrequest_count = self.roundRequestDictionary[i.id][round_day] + self.roundRequestDictionary[i.id][(round_day+1)%7] + self.roundRequestDictionary[i.id][(round_day+2)%7] + self.roundRequestDictionary[i.id][(round_day+3)%7]
                if content_roundrequest_count > storage_content_roundrequest_count:
                    temp_result = 1
        result.append(temp_result)

        temp_result = 0
        if self.search(self.BSList, self.current_full_path[2]).storage.isstored(self.current_requested_content) ==1:
            temp_result = 0 
        else:
            for i in self.search(self.BSList, self.current_full_path[2]).storage.content_storage:
                storage_content_roundrequest_count = self.roundRequestDictionary[i.id][round_day] + self.roundRequestDictionary[i.id][(round_day+1)%7] + self.roundRequestDictionary[i.id][(round_day+2)%7] + self.roundRequestDictionary[i.id][(round_day+3)%7]
                if content_roundrequest_count > storage_content_roundrequest_count:
                    temp_result = 1
        result.append(temp_result)

        temp_result = 0
        if self.dataCenter.storage.isstored(self.current_requested_content) == 1:
            temp_result = 0
        else:
            for i in self.dataCenter.storage.content_storage:
                storage_content_roundrequest_count = self.roundRequestDictionary[i.id][round_day] + self.roundRequestDictionary[i.id][(round_day+1)%7] + self.roundRequestDictionary[i.id][(round_day+2)%7] + self.roundRequestDictionary[i.id][(round_day+3)%7]
                if content_roundrequest_count > storage_content_roundrequest_count:
                    temp_result = 1
        result.append(temp_result)

        return result

    def updatequeue(self):      
        microStartidx=cf.NB_NODES
        bsStartidx=cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1] 
        dcidx=cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1] + cf.NUM_BS[0]*cf.NUM_BS[1]
        coreidx = cf.NB_NODES + cf.NUM_microBS[0]*cf.NUM_microBS[1] + cf.NUM_BS[0]*cf.NUM_BS[1] + 1
        
        #print('ct.updatequeue 에 들어옴')
        #print(path)  
        ty=-1
        if self.path[-1] < microStartidx:
            ty = 0
        elif self.path[-1] < bsStartidx:
            ty = 1
        elif self.path[-1] < dcidx:
            ty = 2
        elif self.path[-1]==dcidx:
            ty = 3
        #print('ty: {}'.format(ty))
        #print('c: {}'.format(c.__dict__))
        #print(network.search(microBSList, path[1]).storage.storage)

        c = self.current_requested_content
        date = self.days[self.round_nb]
        if ty == 1:
            self.search(self.microBSList, self.path[-1]).storage.delContent(c)
            self.search(self.microBSList, self.path[-1]).storage.addContent(c)
            c.update_validity(date)
            #print("MICROBS UPDATE")
            #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))
        if ty == 2:
            self.search(self.BSList, self.path[-1]).storage.delContent(c)
            self.search(self.BSList, self.path[-1]).storage.addContent(c)
            c.update_validity(date)
            #print("BS UPDATE")
            #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))
        if ty == 3:
            self.dataCenter.storage.delContent(c)
            self.dataCenter.storage.addContent(c)
            c.update_validity(date)
            #print("DATACENTER UPDATE")
            #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))

    def printResults(self,ep,episode_reward, random_action_cnt_list, qs_action_cnt_list):
        
        cache_hit_rate = self.total_cache_hit_cnt/self.round_nb
        self.cache_changed_ratio = self.cache_changed_cnt/(self.action_cnt - self.action_3_cnt)   
        avg_hop = self.hop_cnt/self.round_nb

        random_action_cnt = sum(random_action_cnt_list)
        qs_action_cnt = sum(qs_action_cnt_list)

        print('Episode: ', ep+1, '\t', 'Time: ', self.round_nb, '\t', 'action_cnt: ',self.action_cnt,'\t', 'random_action_cnt: ',random_action_cnt,'\t','qs_action_cnt: ',qs_action_cnt,'cache_hit: ', '\t',self.total_cache_hit_cnt, '\t', 'cache_hit_rate: ', cache_hit_rate, '\t', 'avg_hop: ', avg_hop, '\t', 'Reward: ', episode_reward, '\t','cache_changed_ratio: ',self.cache_changed_ratio, '\n')
        
        self.sf.write_result_file(ep, self.round_nb, self.action_cnt, self.total_cache_hit_cnt, cache_hit_rate, avg_hop, episode_reward, self.gamma_episode_reward, self.cache_changed_ratio, self.action_3_cnt, random_action_cnt, qs_action_cnt,
        self.case_A,
        self.case_B,
        self.case_C,
        self.case_D,
        self.case_E,
        self.case_F,
        self.case_G,
        self.case_H,
        self.case_I,
        self.case_J,
        self.case_K,
        self.case_L,
        self.case_M,
        self.case_N,
        self.case_O,
        self.case_P,
        self.case_Q,
        self.case_R,
        )

    def write_result_file(self, ep, time, action_cnt, cache_hit, cache_hit_rate, avg_hop, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt):

        self.sf.write_result_file(ep, time, action_cnt, cache_hit, cache_hit_rate, avg_hop, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt,        
        self.case_A,
        self.case_B,
        self.case_C,
        self.case_D,
        self.case_E,
        self.case_F,
        self.case_G,
        self.case_H,
        self.case_I,
        self.case_J,
        self.case_K,
        self.case_L,
        self.case_M,
        self.case_N,
        self.case_O,
        self.case_P,
        self.case_Q,
        self.case_R)

        self.sf.save_result_file()