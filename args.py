
import argparse
def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", default=False)
    parser.add_argument("--eval_mode", default=False)
    parser.add_argument("--test_mode", default=True)
    parser.add_argument("--model_path", default='mix-con_loss_4_2_graph_mode_without-sib-next-pruning-less-con_loss_only-0.4-x.pt')
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--graphmode", default='astandnext')
    parser.add_argument("--ifedge", default=True)
    parser.add_argument("--whileedge", default=True)
    parser.add_argument("--foredge", default=True) 
    parser.add_argument("--batch_size", default=24)
    parser.add_argument("--num_layers", default=4)
    parser.add_argument("--num_epochs", default=30)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--threshold", default=0)
    parser.add_argument("--eval", default=False)
    parser.add_argument("--mode",default='java-python')
    parser.add_argument("--parse_mode",default='graph_mode')
    parser.add_argument("--maxtree",default=400)
    parser.add_argument("--data_mode",default="more_data")
    parser.add_argument("--aug_mode",default='Delmode')
    parser.add_argument("--aug_multiply",default=1)
    parser.add_argument("--aug_para",default=10)
    parser.add_argument("--aug_ratio",default=0.4)
    parser.add_argument("--all",default=False)

    args = parser.parse_known_args()[0]
    return args
