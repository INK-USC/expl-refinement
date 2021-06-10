import json, pickle, argparse, time
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from Model.data_utils import *
from Model.Phrase_Matcher import *
from Model.Phrase_Matcher_Trainer import *
from Model.Phrase_Matcher_Predictor import *
from Model.Phrase_Matcher_Evaluator import *
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train phrase matcher')
    parser.add_argument('--model', type=str, choices=['fill', 'find'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--save_chk', type=str, default="checkpoint/1.model", help='where to save checkpoint')
    
    # general
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad', type=float, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # model
    parser.add_argument('--encoder', type=str, choices=['bert-mean', 'bert-attn', 'bert-lstm-mean', \
                                                        'bert-lstm-boun', 'bert-lstm-attn', 'bert-baseline'], default='bert-mean')
    parser.add_argument('--output_layer', type=str, choices=['cosine', 'linear', 'bilinear', 'baseline'], default="cosine")
    parser.add_argument('--lstm_hid', type=int, default=100)
    parser.add_argument('--attn_hid', type=int, default=100)
    parser.add_argument('--fine_tune_bert', type=int, default=1)
    parser.add_argument('--train_mode', type=str, choices=['paragraph', 'sent', 'phrase'], default=1)
    
    # data
    parser.add_argument('--neg_pos_ratio', type=float, default=None, help='try to keep a ratio of neg vs. pos samples')
    parser.add_argument('--use_data_em', type=int, default=1, help='(for FIND module) use em data (k query phrases) (total ~23k)')
    parser.add_argument('--n_data_ppdb', type=int, default=None, help='(for FIND module) amount of ppdb data (k pair)')
    
    args = parser.parse_args()
    args = vars(args)
    args['cuda_device'] = torch.device("cuda:"+str(args['gpu']))
    
    ##### Test #####
    # pickle.dump(args, open("./args.p", 'wb'))
    # assert False
    # args = pickle.load(open("./args.p", 'rb'))
    ################
    
    print("\nArguments:")
    for k,v in args.items():
        print(k, ":", v)
    
    print()
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    if args['model'] == 'find':
        if args['train_mode'] == 'paragraph':
            coref_em_data = json.load(open("./Data/train_find.json", 'r')) # ~82k
            coref_em_data = [r for r in coref_em_data if len(r[2]) < 500] # ~82k
            random.shuffle(coref_em_data)
            train, dev_test = split_data(coref_em_data, args['train_mode']) # train: ~90k
            dev = dev_test[:int(len(dev_test)/3)] # ~15k
            test = dev_test[int(len(dev_test)/3):] # ~30k
            if not args['use_data_em']:
                train = filter_em_data(train) # ~67k
            
            train = neg_pos_rebalance(train, args['neg_pos_ratio'], args['train_mode'])
            train = train[:10000]
        elif args['train_mode'] == 'sent':
            coref_em_data = json.load(open("./extract/extract_fill_bert.json", 'r'))
            random.shuffle(coref_em_data)
            train, dev_test = split_data(coref_em_data, args['train_mode'])
            train, dev, test = dev_test[:10000], dev_test[-7000:-5000], dev_test[-5000:]
            # train, dev, test = coref_em_data[:10000], coref_em_data[-7000:-5000], coref_em_data[-5000:]
        else:
            coref_em_data = json.load(open("./Data/train_find_p.json", 'r'))
            random.shuffle(coref_em_data)
            splitted_coref_em_data = []
            for d in coref_em_data:
                for i in range(len(d[0])):
                    splitted_coref_em_data.append([[d[0][i]], d[1], [d[2][i]]])
            
            coref_em_data = splitted_coref_em_data
            train, dev, test = coref_em_data[:20000], coref_em_data[-7000:-5000], coref_em_data[-5000:]
            train = neg_pos_rebalance(train, args['neg_pos_ratio'], args['train_mode'])
            dev = neg_pos_rebalance(dev, args['neg_pos_ratio'], args['train_mode'])
            test = neg_pos_rebalance(test, args['neg_pos_ratio'], args['train_mode'])
        
        model_ = Phrase_Matcher(args)
        
    else:
        pos_data = json.load(open('./Data/train_find_s.json', 'r'))
        c2c_all = [r for rr in pos_data for r in rr]

        random.shuffle(c2c_all)
        c2c_dev, c2c_test = c2c_all[-3500:-2500], c2c_all[-2500:]
        train = c2c_all[:10000]

        train, c2c_dev, c2c_test = [[[d[0], [d[1]], [[d[2]]]] for d in data] for data in [train, c2c_dev, c2c_test]]
        dev = c2c_dev
        test = c2c_test
        
        model_ = Phrase_Matcher(args)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model_.parameters()), lr=args['lr'])
    predictor = Phrase_Matcher_Predictor(args['batch'])
    evaluator = Phrase_Matcher_Evaluator()
    trainer = Phrase_Matcher_Trainer(model_, args, criterion, optimizer, predictor, evaluator)
    trainer.train(train, dev, test)
    
    