import codecs # 对文件读写
import random 
import math 
import numpy as np 
import copy 
import time 

entity2id = {} 
relation2id = {}

# 加载数据集
# 返回实体集合，关系集合，三元组列表。（其中实体集合、关系集合和三元组列表的元素都是实体和关系对应的id）
def data_loader(file): 
    file1 = file + "train.txt" 
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    # 读取实体和关系文件，把实体id文档和关系id文档中的内容装入实体id和关系id字典中，字典结构为{实体:id}
    with open(file2, 'r') as f1, open(file3, 'r') as f2: 
        # 读取整个文件所有行，保存在列表lines1和lines2中，每行作为列表的一个元素，类型为str,结构为'实体 \t id'
        lines1 = f1.readlines() 
        lines2 = f2.readlines()
        for line in lines1: 
            # 按照'\t'符对lines1的每个元素line进行分割，操作之后line变成了一个字符串列表,结构为['实体','id']
            line = line.strip().split('\t') 
            if len(line) != 2:  
                # 进行下一轮循环
                continue 
            # 列表line的实体和id分别作为实体id字典的key和value
            entity2id[line[0]] = line[1] 

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue 
            relation2id[line[0]] = line[1] 

    entity_set = set() 
    relation_set = set()
    triple_list = [] 
    
    # 读取训练集文件，
    with codecs.open(file1, 'r') as f: 
        # 读取训练集文件的所有行，保存在列表content中，每行作为列表的一个元素，类型为str,结构为'头实体 \t 尾实体 \t 关系'
        content = f.readlines() 
        for line in content: 
            # 按照'\t'符对content的每个元素line进行分割，操作之后变成了一个字符串列表triple,结构为['头实体','尾实体','关系']
            triple = line.strip().split("\t")  
            if len(triple) != 3: 
                continue
            # 把实体和关系在id字典中对应的value赋给h_,t_,r_，且这三个值都是id
            h_ = entity2id[triple[0]] 
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]
            # h_,t_,r_组成一个列表，装入triple_list 
            triple_list.append([h_,t_,r_]) 

            entity_set.add(h_) 
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list 
# L2评分函数
def distanceL2(h,r,t): 
    #为方便求梯度，去掉sqrt
    return np.sum(np.square(h + r - t)) 
# L1评分函数
def distanceL1(h,r,t): 
    return np.sum(np.fabs(h+r-t))

# 定义transE模型
class TransE:
    # 初始化模型需要的参数
    def __init__(self,entity_set, relation_set, triple_list, 
                 embedding_dim=100, learning_rate=0.01, margin=1,L1=True):
        # embedding维度
        self.embedding_dim = embedding_dim
        # 学习率
        self.learning_rate = learning_rate
        # 边缘参数（正负样本三元组之间的间隔修正）
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1=L1
        self.loss = 0

    # 生成实体和关系的初始embedding，将实体id集合、关系id集合转变为{实体id：实体向量}、{关系id：关系向量}这两个字典
    def emb_initialize(self): 
        relation_dict = {}
        entity_dict = {}

        # 对关系向量进行初始化
        for relation in self.relation: 
            # 生成100维的向量，每一维的取值随机，范围在[-0.6~0.6)之间
            r_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) , 
                                           6/math.sqrt(self.embedding_dim) ,
                                           self.embedding_dim)
            # 向量正则化后与对应id组合成字典中的一个元素
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp,ord=2) 

        # 对实体向量进行初始化
        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) ,
                                        6/math.sqrt(self.embedding_dim) ,
                                        self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp,ord=2)

        self.relation = relation_dict
        self.entity = entity_dict

    # 定义训练函数
    def train(self, epochs):
        # 数据集分为40个batch来训练
        nbatches = 40 
        # 每个batch的大小
        batch_size = len(self.triple_list) // nbatches 
        print("batch size: ", batch_size)
        for epoch in range(epochs): 
            start = time.time() 
            self.loss = 0 #

            for k in range(nbatches): 
                # 从triple_list中随机采样batch_size数量的三元组组成Sbatch，Sbatch类型为列表，元素类型为[h_,r_,t_]
                Sbatch = random.sample(self.triple_list, batch_size) 
                # 用来存放元组对（原三元组，负三元组）的列表
                Tbatch = [] 

                for triple in Sbatch: 
                    # 生成负三元组
                    corrupted_triple = self.Corrupt(triple) 
                    # 如果正负样本组成的元组对不在Tbatch中，则将其放入Tbatch列表中
                    if (triple, corrupted_triple) not in Tbatch: 
                        Tbatch.append((triple, corrupted_triple))  
                # 调用update_triple_embedding函数，计算这一个batch的损失值，根据梯度下降法更新向量，然后再进行下一个batch的训练
                self.update_embeddings(Tbatch) 

            end = time.time() 
            # 记录该轮的信息
            print("epoch: ", epoch , "cost time: %s"%(round((end - start),3))) 
            print("loss: ", self.loss)

            
        # 所有的40个batch训练完成后，将训练好的实体向量、关系向量输出到目录下
        print("写入文件...")
        with codecs.open("example_result//entity_50dim_batch40", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("example_result//relation_50dim_batch40", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")

    # 定义生成三元组负样本的函数
    def Corrupt(self,triple): 
        # deepcopy将复制对象完全复制一遍，并作为一个独立的新个体单元存在。即使改变被复制对象，deepcopy新个体也不会发生变化
        corrupted_triple = copy.deepcopy(triple) 
        seed = random.random() 
        if seed > 0.5: 
            # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:
                # 随机采样一个新的头实体
                # rand_head = random.sample(self.entity.keys(),1)[0] 
                rand_head = random.sample(sorted(self.entity.keys()), 1)[0] 
            corrupted_triple[0]=rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                # 随机采样一个新的尾实体
                # rand_tail = random.sample(self.entity.keys(), 1)[0] 
                rand_tail = random.sample(sorted(self.entity.keys()), 1)[0] 
            corrupted_triple[1] = rand_tail
        return corrupted_triple 

    # 定义更新函数 
    def update_embeddings(self, Tbatch): 
        # 调用deepcopy函数深拷贝实体字典和关系字典 
        copy_entity = copy.deepcopy(self.entity) 
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch: 
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]
            # 根据L1范数或L2范数计算得分，计算三元组的距离
            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            # 计算损失
            err = self.hinge_loss(dist_correct, dist_corrupt) 

            if err > 0:
                self.loss += err
                # 计算L2评分函数的梯度
                grad_pos = 2 * (h_correct + relation - t_correct) 
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                # 如果使用L1评分函数进行梯度更新，L1评分函数的梯度向量中每个元素为-1或1
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                
                # 两个三元组只有一个entity不同（不同时替换头尾实体），所以在每步更新时重叠实体需要更新两次（和更新relation一样）。
                # 例如正确的三元组是（1，2，3），错误的是（1，2，4），那么1和2都需要更新两次，针对正确的三元组更新一次，针对错误的三元组更新一次
                # 若替换的是尾实体，则头实体更新两次
                if triple[0] == corrupted_triple[0]:  
                    # corrupt项整体为负，因此符号与correct相反
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg
                # 若替换的是头实体，则尾实体更新两次
                elif triple[1] == corrupted_triple[1]:  
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                # relation更新两次
                relation_update -= self.learning_rate*grad_pos
                relation_update -= (-1)*self.learning_rate*grad_neg


        # batch norm
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i]) 
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        self.entity = copy_entity
        self.relation = copy_relation

    # 定义损失函数
    def hinge_loss(self,dist_correct,dist_corrupt):
        return max(0,dist_correct-dist_corrupt+self.margin)


if __name__=='__main__':
    # 加载数据集
    file1 = "data//" 
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (len(entity_set),len(relation_set),len(triple_list)))
    # 训练
    transE = TransE(entity_set, relation_set, triple_list,embedding_dim=50, learning_rate=0.01, margin=1,L1=True)
    transE.emb_initialize()
    transE.train(epochs=20)
