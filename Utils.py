import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import time

def get_evaluation(prediction,labels,args):
    index = 0
    tp=0
    tn=0
    fp=0
    fn=0
    for pre in prediction:
        if pre>args.threshold and labels[index]==1:
            tp+=1
        if pre<=args.threshold and labels[index]==-1:
            tn+=1
        if pre>args.threshold and labels[index]==-1:
            fp+=1
        if pre<=args.threshold and labels[index]==1:
            fn+=1  
    a=0.0
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        p = 0
    else:
        p=tp/(tp+fp)
    if tp+fn==0:
        r=0
    else:
        r=tp/(tp+fn)
    a=(tp+tn)/(tp+tn+fp+fn)
    if(p+r==0):
        f1=0
    else:
        f1=2*p*r/(p+r)
    return a,p,r,f1
def save_v(filename,v,base_path='./'):
    f1 = open(base_path+filename, 'wb')
    pickle.dump(v, f1)
    f1.close()
    print(filename, 'Save Over!')
def open_v(filename,base_path='./'):
    f = open(base_path+filename, 'rb')
    r = pickle.load(f)
    f.close()
    print(filename,'Load Over!')
    return r 

def test(dataset,device,model,mode):
    model.eval()
    data_count=0
    results=[]
    cos_right = []
    cos_wrong = []
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    with torch.no_grad():
        for data_indice in dataset:
            data_count+=len(data_indice)
            graph_vec1 = []
            graph_vec2 = []
            taskids = []
            for data,label,taskid in data_indice:
                label=torch.tensor(label, dtype=torch.float, device=device)
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
                graph_vec1.append(prediction[0])
                graph_vec2.append(prediction[1])
                taskids.append(taskid)
            for i in range(len(graph_vec1)):
                cos_right.append(nn.CosineSimilarity(dim=1)(graph_vec1[i],graph_vec2[i])) 
            for i in range(len(graph_vec1)):
                nag_count = 0
                for j in range(len(graph_vec1)):
                    if i==j:
                        continue
                    if taskids[i]==taskids[j]:
                        continue
                    cos_wrong.append(nn.CosineSimilarity(dim=1)(graph_vec1[i],graph_vec1[j]))
                    break
    print(data_count,len(cos_wrong),len(cos_right))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    temp_best_f1 = 0
    temp_best_recall = 0
    temp_best_precision = 0
    temp_best_threshold = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in tqdm(range(1,100)):
        count = 0
        error_count = 0
        threshold = i/100
        for m in cos_right:
            if m >= threshold:
                count+=1
                tp+=1
            else:
                fp+=1
        total = len(cos_right)
        for n in cos_wrong:
            if n < threshold:
                error_count += 1
                tn+=1
            else:
                fn+=1
        error_total = len(cos_wrong)
        correct_recall = count/total
        if error_total-error_count+count == 0:
            continue
        precision = count/(error_total-error_count+count) 
        if precision+correct_recall == 0:
                    continue
        F1 = 2*precision*correct_recall/(precision+correct_recall)
        if F1 > temp_best_f1:
            temp_best_f1 = F1
            temp_best_recall = correct_recall
            temp_best_precision = precision
            temp_best_threshold = threshold
    
    results = {'mode': mode,'precision': temp_best_precision,'recall': temp_best_recall,  'F1': temp_best_f1,
              'threshold': temp_best_threshold}

    return results