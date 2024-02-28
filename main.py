import sys
print(sys.path[0])
from Utils import open_v,test
from tqdm import tqdm, trange
from args import get_options
options = get_options()
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(sys.path[0]+'\log\{}.log'.format(options.mode))
formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formats)
logger.addHandler(fh)


print(options.cuda)
logger.info(options)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
device=torch.device('cuda:0')


data_path ='./jsondata/pair_train.jsonl'
logger.info("Path:%s",data_path)

model_path = './saved_model/mix-con_loss_4_2_graph_mode_without-sib-next-pruning-less-con_loss_only-0.4-x.pt'

logger.info("Model Path:%s",model_path)
train_iter = open_v('/saved_var/train_iter-mix-{}-{}-{}-{}-{}-pruning-less-0.4-full'.format(options.maxtree,options.parse_mode,options.data_mode,options.aug_para,options.aug_multiply))
vocablen = open_v('/saved_var/vocblen-mix-{}-{}-{}-{}-{}-pruning-less-0.4-full'.format(options.maxtree,options.parse_mode,options.data_mode,options.aug_para,options.aug_multiply))
eval_iter = open_v('/saved_var/eval_iter-mixlang-{}-{}-{}-{}-{}-pruning-less-0.4-full'.format(options.maxtree,options.parse_mode,options.data_mode,options.aug_para,options.aug_multiply))




logger.info("***** Running model initialization *****")
import models
num_layers=int(options.num_layers)
model=models.GMNnet(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total Parameters:'+str(total_num)) 
print('Trainable Parameters'+str(trainable_num))
if options.train_mode:
    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    criterion=nn.CosineEmbeddingLoss()
    criterion2=nn.MSELoss()
    model.train()
    logger.info("***** Running model training *****")

    tau = 10
    temp_best_F1 = 0
    epochs = trange(options.num_epochs, leave=True, desc = "Epoch")
    for epoch in epochs:
        print(epoch)
        totalloss=0.0
        main_index=0.0
        for index, batch in tqdm(enumerate(train_iter), total=len(train_iter), desc = "Batches"):
            optimizer.zero_grad()
            posloss= 0
            graph_group_1 = []
            graph_group_2 = []
            taskid_list = []
            for data,label,taskid in batch:
                label=torch.tensor(1, dtype=torch.float, device=device)
                labelneg=torch.tensor(-1, dtype=torch.float, device=device)
                taskid_list.append(taskid)
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
                x1=torch.tensor(x1, dtype=torch.long, device=device)
                x2=torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
                edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
                if edge_attr1!=None:
                    edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
                    edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
                data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
                prediction=model(data)
                cossim=F.cosine_similarity(prediction[0],prediction[1])
                graph_vec_1,graph_vec_2 = prediction[0],prediction[1]
                graph_group_1.append(graph_vec_1)
                graph_group_2.append(graph_vec_2)
                posloss=posloss+criterion2(cossim,label)
            
            loss_temp = torch.zeros((len(graph_group_1),len(graph_group_1)*2-1),device=device,dtype=torch.float)
            negloss = 0
            neg_count=0
            for i in range(len(graph_group_1)):
                loss_temp[i][0] = (nn.CosineSimilarity(dim=1)(graph_group_1[i],graph_group_2[i]) + 1) * 0.5 * tau
                indice = 1
                for j in range(len(graph_group_1)):
                    if i==j:
                        continue
                    temp = j
                    while taskid_list[i]==taskid_list[temp]:
                        temp = (temp + 1) % (len(graph_group_1))
                    loss_temp[i][indice] = (nn.CosineSimilarity(dim=1)(graph_group_1[i],graph_group_2[temp]) + 1) *0.5 * tau
                    cossim=F.cosine_similarity(graph_group_1[i],graph_group_2[temp])
                    negloss=negloss+criterion2(cossim,labelneg)
                    indice += 1
                    loss_temp[i][indice] = (nn.CosineSimilarity(dim=1)(graph_group_1[i],graph_group_1[temp]) + 1) * 0.5 * tau
                    cossim=F.cosine_similarity(graph_group_1[i],graph_group_1[temp])
                    negloss=negloss+criterion2(cossim,labelneg)
                    indice += 1
                neg_count += (indice-1)
            con_loss = -torch.nn.LogSoftmax(dim=1)(loss_temp)
            con_loss = torch.sum(con_loss, dim=0)[0]
            con_loss = con_loss / len(graph_group_1)
            batchloss = (posloss + negloss)/len(graph_group_1)
            negloss = negloss/neg_count
            posloss = posloss/(len(graph_group_1))
            loss = con_loss + negloss+ posloss
            tr_loss = loss
            loss.backward()    
            optimizer.step()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item(),8))
        logger.info("  Num epoch = %d \n", epoch)
        logger.info(" Total loss= %g \n",round(loss.item(),8))
        if options.eval_mode:
            result = test(eval_iter,device,model,options.mode)
            logger.info(result)
            if result.F1> temp_best_F1:
                temp_best_F1 = result.F1
                logger.info("***** Saving model *****")
                torch.save(model.state_dict(), model_path)
        else:
            logger.info("***** Saving model *****")
            torch.save(model.state_dict(), model_path)
if options.test_mode:
    model=models.GMNnet(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)
    model_path = sys.path[0]+'/saved_model/'+ options.model_path
    model.load_state_dict(torch.load(model_path))
    logger.info("***** Running model initialization *****")
    logger.info("Test Model Path:%s",model_path)
    model.eval()
    test_mode = ['java-python','java-cpp','java-cs','python-cpp','python-cs','cpp-cs']
    avg_p = {'java-python':0,'java-cpp':0,'java-cs':0,'python-cpp':0,'python-cs':0,'cpp-cs':0}
    avg_r = {'java-python':0,'java-cpp':0,'java-cs':0,'python-cpp':0,'python-cs':0,'cpp-cs':0}
    avg_f = {'java-python':0,'java-cpp':0,'java-cs':0,'python-cpp':0,'python-cs':0,'cpp-cs':0}
    for i in range(5):
        result = []
        for data_mode in test_mode:
            test_iter = open_v('/saved_var/test/{}/test_iter-{}-{}-{}-more_edge-new-test-online-p-{}-full-{}'.format(options.aug_ratio,data_mode,options.maxtree,options.parse_mode,options.aug_ratio,i))
            result.append(test(test_iter,device=device,model=model,mode=data_mode)) 
        logger.info("***** Result *****")
        for item in result:
            logger.info(item)
            avg_p[item['mode']]+=item['precision']
            avg_r[item['mode']]+=item['recall']
            avg_f[item['mode']]+=item['F1']
    for data_mode in test_mode:
        avg_p[data_mode] = round(avg_p[data_mode]/5,8)
        avg_r[data_mode] = round(avg_r[data_mode]/5,8)
        avg_f[data_mode] = round(avg_f[data_mode]/5,8)
    logger.info(avg_p)
    logger.info(avg_r)
    logger.info(avg_f)
    avg_p = 0
    avg_r = 0
    avg_f = 0
    for i in range(5):
        test_iter = open_v('/saved_var/test/{}/test_iter-mixlang-{}-{}-more_edge-new-test-online-p-{}-full-{}'.format(options.aug_ratio,options.maxtree,options.parse_mode,options.aug_ratio,i))
        result = test(test_iter,device=device,model=model,mode='fullmode')
        logger.info("***** Total Result *****") 
        logger.info(result)
        avg_p+=result['precision']
        avg_r+=result['recall']
        avg_f+=result['F1']
    logger.info("Precision:%g",round(avg_p/5,8))
    logger.info("recall:%g",round(avg_r/5,8))
    logger.info("F1:%g",round(avg_f/5,8))

