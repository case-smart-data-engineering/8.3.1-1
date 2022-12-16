import numpy as np
import codecs # 对文件读写
import operator # operator模块定义了与内置运算相对应的函数，如算术操作、比较操作以及和标准API相对应的操作。
import json 
from my_solution import data_loader,entity2id,relation2id

# # 加载数据集,生成实体字典，关系字典，测试三元组
def dataloader(entity_file,relation_file,test_file): 
    
    entity_dict = {} 
    relation_dict = {}
    test_triple = [] 

    # 生成实体字典，结构为{实体id:向量}
    with codecs.open(entity_file) as e_f: 
        lines = e_f.readlines() 
        for line in lines:
            # 将每一条数据分为entity和embedding
            entity,embedding = line.strip().split('\t') 
            # 将embedding转为Python对象
            embedding = json.loads(embedding) 
            # 把embedding和对应的entity作为实体字典一个元素
            entity_dict[entity] = embedding 
    # 生成关系字典，结构为{关系id:向量}
    with codecs.open(relation_file) as r_f: 
        lines = r_f.readlines()
        for line in lines:
            relation,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding
    # 生成测试三元组，结构为[(h_,t_,r_),...]
    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            # 把测试集的数据拆分成后装入列表
            triple = line.strip().split('\t') 
            if len(triple) != 3:
                continue
            # 把头实体对应的id赋给h_
            h_ = entity2id[triple[0]] 
            # 把尾实体对应的id赋给t_
            t_ = entity2id[triple[1]]
            # 把关系实体对应的id赋给r_
            r_ = relation2id[triple[2]]
            # 把由h_,t_,r_组成的元组装入测试三元组
            test_triple.append(tuple((h_,t_,r_))) 

    return entity_dict,relation_dict,test_triple

# 定义评分函数
def distance(h,r,t): 
    # np.array()的作用就是把列表转化为n维数组(向量)
    h = np.array(h) 
    r=np.array(r)
    t = np.array(t)
    # 计算差值向量
    s=h+r-t 
    return np.linalg.norm(s) 

# 定义测试模型
class Test:
    # 初始化模型需要的参数
    def __init__(self,entity_dict,relation_dict,test_triple,train_triple,isFit = True): 
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit 
        # 排名中正确预测排在前10位的概率（命中前10的次数/总查询次数，越大越好）
        self.hits10 = 0 
        # 排名中的正确预测所在的排名的平均值（正确结果排名之和/总查询次数，越小越好）
        self.mean_rank = 0 



        self.relation_hits10 = 0
        self.relation_mean_rank = 0 
    # 定义实体排名函数
    def rank(self):
        hits = 0 
        rank_sum = 0 
        step = 1

        for triple in self.test_triple: 
            # 排名字典，类型为{(id,id,id):评分}
            rank_head_dict = {} 
            rank_tail_dict = {}

            for entity in self.entity_dict.keys(): 
                # 生成更换头实体后的负三元组，由id组成
                corrupted_head = [entity,triple[1],triple[2]] 
                if self.isFit: 
                    # 如果生成的负三元组没在训练集中，就取出该负三元组的三个id对应的向量
                    if corrupted_head not in self.train_triple: 
                        h_emb = self.entity_dict[corrupted_head[0]] 
                        r_emb = self.relation_dict[corrupted_head[2]]
                        t_emb = self.entity_dict[corrupted_head[1]]
                        # 计算得分，并生成头实体排名字典的元素
                        rank_head_dict[tuple(corrupted_head)]=distance(h_emb,r_emb,t_emb) 
                
                else:
                    h_emb = self.entity_dict[corrupted_head[0]]
                    r_emb = self.relation_dict[corrupted_head[2]]
                    t_emb = self.entity_dict[corrupted_head[1]]
                    rank_head_dict[tuple(corrupted_head)] = distance(h_emb, r_emb, t_emb)
            
                # 生成更改尾实体后的负三元组
                corrupted_tail = [triple[0],entity,triple[2]]
                if self.isFit:
                    if corrupted_tail not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_tail[0]]
                        r_emb = self.relation_dict[corrupted_tail[2]]
                        t_emb = self.entity_dict[corrupted_tail[1]]
                        rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_tail[0]]
                    r_emb = self.relation_dict[corrupted_tail[2]]
                    t_emb = self.entity_dict[corrupted_tail[1]]
                    
                    rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)
            
            
            # 对排名字典按照评分进行从小到大排序
            rank_head_sorted = sorted(rank_head_dict.items(),key = operator.itemgetter(1)) 
            rank_tail_sorted = sorted(rank_tail_dict.items(),key = operator.itemgetter(1))

            
            
            # 统计正确头实体的排名,累计hits和rank_sum
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]: 
                    if i<10:
                        hits += 1 
                    rank_sum = rank_sum + i + 1 
                    break
            # 统计正确尾实体的排名,累计hits和rank_sum
            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i<10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            step += 1 
            if step % 100 == 0: 
                print("step ", step, " ,hits ",hits," ,rank_sum ",rank_sum)
                print()
        
        # 因为头尾实体各比较一次，所以一个三元组会统计两次hits和rank_sum
        self.hits10 = hits / (2*len(self.test_triple)) 
        self.mean_rank = rank_sum / (2*len(self.test_triple))
    # 定义关系排名函数
    def relation_rank(self):  
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            # 关系字典，元素类型为{id:得分}
            rank_dict = {} 
            for r in self.relation_dict.keys(): 
                # 生成负三元组
                corrupted_relation = (triple[0],triple[1],r) 
                # 若负三元组在训练集中，则继续生成负三元组
                if self.isFit and corrupted_relation in self.train_triple:
                    continue
                # 对于不在训练集中的负三元组，依次取出向量
                h_emb = self.entity_dict[corrupted_relation[0]] 
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                # 计算得分，并装入排名字典，元素类型尾{关系id:评分}
                rank_dict[r]=distance(h_emb, r_emb, t_emb) 
            # 对排名字典按照评分进行从小到大排序
            rank_sorted = sorted(rank_dict.items(),key = operator.itemgetter(1)) 
            # rank用来统计预测正确的排名
            rank = 1 
            # 统计正确关系的排名,累计hits和rank_sum  
            for i in rank_sorted:
                # 预测正确则退出循环
                if triple[2] == i[0]: 
                    break 
                # 预测错误则继续比较下一个项
                rank += 1 
            if rank<10:
                hits += 1
            rank_sum = rank_sum + rank + 1

            step += 1
            if step % 100 == 0:
                print("relation step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
                print()
        # 计算关系的hits10和mean_rank
        self.relation_hits10 = hits / len(self.test_triple) 
        self.relation_mean_rank = rank_sum / len(self.test_triple)

if __name__ == '__main__':
    # 加载测试数据集
    _, _, train_triple = data_loader("test_data//") 
    # 生成实体字典，关系字典，测试三元组
    entity_dict, relation_dict, test_triple = \
        dataloader("test_result//entity_50dim_batch40","test_result//relation_50dim_batch40",
                   "test_data//test.txt")

    # 初始化测试实例
    test = Test(entity_dict,relation_dict,test_triple,train_triple,isFit=False)
    # 调用rank函数计算实体的hits10和mean_rank
    test.rank()
    print("entity hits@10: ", test.hits10)
    print("entity meanrank: ", test.mean_rank)

    # 调用rank函数计算关系的hits10和mean_rank
    test.relation_rank()
    print("relation hits@10: ", test.relation_hits10)
    print("relation meanrank: ", test.relation_mean_rank)
    # 保存结果
    f = open("test_result//outcome.txt",'w')
    f.write("entity hits@10: "+ str(test.hits10) + '\n')
    f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    f.close()

