import math
from platform import node

# Scenario Generating
CONTENT_FILE = "eMBB"
GEN_METHOD = "gaussian"

# for network current caching algorithm
allocation_algorithm = None # LCE, LCD, RND
replacement_algorithm = None # FIFO, LRU, LFU


# Episode Number
MAX_EPISODE_NUM = 2000
TOTAL_PRIOD = 24 #24 #(week)
MAX_ROUNDS = 5000  #70000 2만이나 3만으로 늘리고 기간을 한달이라고 하면 총 24 * 30일 
MAX_REQ_PER_ROUND = 1

# info of node
NB_NODES = 300 # 300개
TX_RANGE = 30 # meters

# area definition
AREA_WIDTH = 5000.0
AREA_LENGTH = 5000.0

# Basestation configuration
NUM_microBS = [3,3] # 36개
NUM_BS = [2,2] # 9개
DATACENTER_ID = NB_NODES + NUM_microBS[0]*NUM_microBS[1] + NUM_BS[0]*NUM_BS[1]
CORE_ID = DATACENTER_ID + 1

# Basestation idx info
microStartidx = NB_NODES
bsStartidx = microStartidx + NUM_microBS[0]*NUM_microBS[1] 
dcidx = bsStartidx + NUM_BS[0]*NUM_BS[1]
coreidx = dcidx + 1


# storage size
CONTENT_SIZE = 20
microBS_SIZE = 100
BS_SIZE = 100
CENTER_SIZE = 200
#scenario info




#잠깐 정리
# network latency = propagation delay + transmission delay + processing delay 
# #https://www.rfwireless-world.com/calculators/Network-Latency-Calculator.html
# propagation delay = distance / speed
# transmission delay = packet size (bits)/throughput
# serialization delay = packet size (bits) / Transmission Rate (bps)

#https://5g-tools.com/5g-nr-throughput-calculator/
mu = 30
BW = 50

# J : number of aggregated component carriers, maximum number (3GPP 38.802): 16
J = 1

# maximum number of MIMO layers, 3GPP 38.802: maximum 8 in DL, maximum 4 in UL
v_layers = 4

# modulation order
Q_m =  6

# scaling factor
f = 1

# R_max : Target code Rate R / 1024
R_max = 0.92578125

# maximum # of PRB
# 3GPP 38.213 Table 5.3.2-1 Transmission bandwidth configuration NRB for FR1
# BW = 50MHz , FR1 = 30kHz --> NRB = 133
N_bwPRB = 133

# average OFDM symbol duration in a subframe
t_us = math.pow(10, -3)/(14*math.pow(2,1))

# overhead for control channels
UL_OH = 0.08
DL_OH = 0.14

# TDD throughput(bps)
ULthroughput = J * v_layers * Q_m * f * R_max * (N_bwPRB*12)/ t_us * (1 - UL_OH) * 0.214
DLthroughput = J * v_layers * Q_m * f * R_max * (N_bwPRB*12)/ t_us * (1 - DL_OH) * 0.771

DLpackets_per_second = 108281.25
ULpackets_per_second = 29687.5

LIGHT_SPEAD = 299792458
PACKET_SIZE = 12800 #(1500byte)

#packet size embb 1500byte
#packet size urllc 32bytes
#pacekt size mMTC 32bytes
#throughput https://5g-tools.com/5g-nr-throughput-calculator/

LATENCY_INTERNET = 0.0025#0.0422 #ms



#DQN parameters

# DQN structure
DROPOUT_RATE = 0.2
H1 = 1
H2 = 3
H3 = 9
H4 = 9
H5 = 9
H6 = 6
H7 = 4
H8 = 1
q = 4

# DQN 하이퍼파라미터
GAMMA = 0.95 #0.95
BATCH_SIZE = 1024 #64
BUFFER_SIZE = 100000
LEARNING_RATE = 0.000005
TAU = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.20
EPSILON_FIN_MIN = 0.01
#action
NB_ACTION = 70

#DELETE PERIOD
DELETE_PERIOD = 14


# reward parameter
a = 1.5 # 1
b = 2 #0.2
c = 0 #0.5
d = 0 #0.1
e = 0 # 10

# Twin Network parameter
TWIN_ROUND = 15

P_REWARD = 24 #18 #46
N_REWARD = -18
