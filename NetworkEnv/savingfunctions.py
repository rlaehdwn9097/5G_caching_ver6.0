import matplotlib.pyplot as plt
import NetworkEnv.config as cf
from time import strftime, localtime, time
import os
import csv

class savingfunctions():

    def __init__(self):

        tm = localtime(time())
        self.Date = strftime('%Y-%m-%d_%H-%M-%S', tm)

        if os.path.isdir('LabResults'):
            pass
        else:
            os.mkdir('LabResults')

        self.folderName = "LabResults/" + str(self.Date)
        os.mkdir(self.folderName)

        self.index=1
        self.set_result_file()

        self.save_config_file()
        
        #self.tmp_file = open(self.folderName +"/tmp.txt",'a', encoding="UTF-8")
        #self.actionDictionary_file = open(self.folderName +"/actionDictionary.txt",'a', encoding="UTF-8")

    
    ## save them to file if done
    def plot_reward_result(self, save_epi_reward):
        plt.plot(save_epi_reward)
        plt.savefig(self.folderName + '/rewards.png')
        plt.show()

    ## save them to file if done
    def plot_cache_hit_result(self, save_epi_cache_hit_rate):
        plt.plot(save_epi_cache_hit_rate)
        plt.savefig(self.folderName + '/cache_hit_rate.png')
        plt.show()

    def plot_redundancy_result(self, save_epi_redundancy):
        plt.plot(save_epi_redundancy)
        plt.savefig(self.folderName + '/redundancy.png')
        plt.show()

    def plot_existing_content_result(self, existing_content):
        plt.plot(existing_content)
        plt.savefig(self.folderName + '/existing_content.png')
        plt.show()

    def plot_denominator_result(self, denominator):
        plt.plot(denominator)
        plt.savefig(self.folderName + '/denominator.png')
        plt.show()

    def plot_avg_hop_result(self, avg_hop):
        plt.plot(avg_hop)
        plt.savefig(self.folderName + '/avg_hop.png')
        plt.show()


    def write_tmp_reward(self, d_core, d_cache, c_node, vancancy, front, back):
        
        #self.tmp_file.write("============ EPSIODE : {} ============\n".format(+1))
        for i in range(len(front)):
            self.tmp_file.write("{}번째\n".format(i))
            self.tmp_file.write("reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) - self.c*self.alpha_redundancy - self.d*self.beta_redundancy - self.e*self.vacancy\n")
            self.tmp_file.write("d_core : {}\n".format(d_core[i]))
            self.tmp_file.write("d_cache : {}\n".format(d_cache[i]))
            self.tmp_file.write("c_node : {}\n".format(c_node[i]))
            self.tmp_file.write("alpha_redundancy : {}\n".format(alpha_redundancy[i]))
            self.tmp_file.write("beta_redundancy : {}\n".format(beta_redundancy[i]))
            self.tmp_file.write("vancancy : {}\n".format(vancancy[i]))
            self.tmp_file.write("front : {}\n".format(front[i]))
            self.tmp_file.write("back : {}\n\n".format(back[i]))


    def write_actionDictionary_file(self, episode, requestDictionary, actionDictionary):
        """
        actionDictionary = dict(sorted(actionDictionary.items(), key=lambda x:len(x[1]), reverse=True))
        actionDictionary_keys = actionDictionary.keys()
        self.actionDictionary_file.write("Episode : {}\n".format(episode))
        for title in actionDictionary_keys:
            self.actionDictionary_file.write("{} : ".format(title))
            self.actionDictionary_file.write("{}".format(actionDictionary[title]))
            self.actionDictionary_file.write("\n")
        """
        requestDictionary = dict(sorted(requestDictionary.items(), key=lambda x:x[1], reverse=True))
        requestDictionaryKeys = requestDictionary.keys()

        self.actionDictionary_file.write("Episode : {}\n".format(episode))
        for title in requestDictionaryKeys:
            self.actionDictionary_file.write("{} : ".format(title))
            self.actionDictionary_file.write("{}".format(actionDictionary[title]))
            self.actionDictionary_file.write("\n")

    def add_index(self):
        self.index += 1
        print(self.index)

    def set_result_file(self):
        #print("set_result_file 들어옴")
        self.result_file = open(self.folderName +"/result{}.csv".format(self.index),'a', newline='')
        self.result_file_writer = csv.writer(self.result_file)
        self.result_file_writer.writerow(['ep', 'time', 'action_cnt', 'cache_hit', 'cache_hit_rate', 'avg_hop', 'episode_reward', 'gamma_episode_reward', 'cache_changed_cnt', 'action_3_cnt', 'random_action_cnt', 'qs_action_cnt'])

    def save_result_file(self):
        self.result_file.close()
        self.result_file = open(self.folderName +"/result{}.csv".format(self.index),'a', newline='')
        self.result_file_writer = csv.writer(self.result_file)

    def write_result_file(self, ep, time, action_cnt, cache_hit, cache_hit_rate, avg_hop, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r):
        
        row = [ep+1, time, action_cnt, cache_hit, cache_hit_rate, avg_hop, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
        self.result_file_writer.writerow(row)
        

    def set_meta_file(self):

        self.meta_file = open(self.folderName +"/metafile.txt",'a')
        self.write_meta_file()

    def save_config_file(self):
        with open('./NetworkEnv/config.py', "r",encoding='UTF8') as source, open(self.folderName +"/config.py", "w",encoding='UTF8') as destination:
            content = source.read()
            destination.write(content)

     
