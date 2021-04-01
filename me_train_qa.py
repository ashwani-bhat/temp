import numpy as np
import json, os, logging, pickle, argparse
from evaluate_squad import compute_f1
# from simpletransformers.question_answering import QuestionAnsweringModel
from arch import QuestionAnsweringModel

# code taken from 
# https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], []
    
    for i, key in enumerate(['correct_text', 'similar_text', 'incorrect_text']):
        for item in text[key]:
            if i==0:
                if 'impossible' in item and text[key][item]['predicted'] == '':
                    impos1 += 1
                elif 'span' in item:
                    pos1 += 1
                    fscores.append(1)
                    squad_fscores.append(1)
                    
            elif i==1:
                if 'impossible' in item:
                    impos2 += 1
                elif 'span' in item:
                    
                    z = text[key][item]
                    if z['predicted'] != '':
                        longest_match = list(lcs(z['truth'], z['predicted']))[0]
                        lcs_all.append(longest_match)
                        partial_match_scores.append(round(len(longest_match.split())/len(z['truth'].split()), 4))
                        pos2 += 1
                        r = len(longest_match.split())/len(z['truth'].split())
                        p = len(longest_match.split())/len(z['predicted'].split())
                        f = 2*p*r/(p+r)
                        fscores.append(f)
                        squad_fscores.append(compute_f1(z['truth'], z['predicted']))
                    else:
                        pos3 += 1
                        impos4 += 1
                        fscores.append(0)
                        squad_fscores.append(0)
                                
                    
            if i==2:
                if 'impossible' in item:
                    impos3 += 1
                elif 'span' in item:
                    if z['predicted'] == '':
                        impos4 += 1
                    pos3 += 1
                    fscores.append(0)
                    squad_fscores.append(0)
                    
    total_pos = pos1 + pos2 + pos3
    imr = impos2/(impos2+impos3)
    imp = impos2/(impos2+impos4)
    imf = 2*imp*imr/(imp+imr)
    
    p1 = 'Postive Samples:'
    p2 = 'Exact Match: {}/{} = {}%'.format(pos1, total_pos, round(100*pos1/total_pos, 2))
    p3 = 'Partial Match: {}/{} = {}%'.format(pos2, total_pos, round(100*pos2/total_pos, 2))
    p4a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores), 2))
    p4b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores), 2))
    p5 = 'No Match: {}/{} = {}%'.format(pos3, total_pos, round(100*pos3/total_pos, 2))
    p6 = '\nNegative Samples'
    p7 = 'Inv F1 Score = {}%'.format(round(100*imf, 2))
    # p7a = 'Inv Recall: {}/{} = {}%'.format(impos2, impos2+impos3, round(100*imr, 2))
    # p7b = 'Inv Precision: {}/{} = {}%'.format(impos2, impos2+impos4, round(100*imp, 2))
    
    p = '\n'.join([p1, p2, p3, p4a, p4b, p5, p6, p7])
    return p


if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate') 
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob | span')
    parser.add_argument('--fold', type=int, default=1, metavar='F', help='which fold')
    parser.add_argument('--context', action='store_true', default=False, help='use context')
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    parser.add_argument('--doc_stride', type=int, default=512, metavar='D', help='chunk strides')
    parser.add_argument('--slide', action='store_true', default=False, help='sliding window')
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False, help='overwrite output')
    
    args = parser.parse_args()

    print(args)
    
    model_family = {
        'rob': 'roberta', 
        'span': 'bert', 
        'spanbi': 'bertbi',
        'xlnet': 'xlnet', 
        'longformer': 'longformer',
        'bert': 'bert',
        'bertbi': 'bertbi',
    }
    
    model_id = {
        'rob': 'roberta-base', 
        'span': 'spanbert-squad',
        'spanbi': 'spanbert-bilstm', 
        'xlnet': 'xlnet-base-cased', 
        'longformer': 'allenai/longformer-base-4096',
        'bert': 'bert-base-uncased',
        'bertbi': 'bert-base-uncased-bilstm',
    }

    model_exact_id = {
        'rob': 'roberta-base', 
        'span': 'mrm8488/spanbert-finetuned-squadv2', 
        'spanbi': 'mrm8488/spanbert-finetuned-squadv2',
        'xlnet': 'xlnet-base-cased', 
        'longformer': 'allenai/longformer-base-4096',
        'bert': 'bert-base-uncased',
        'bertbi': 'bert-base-uncased',
    }
    
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = args.model
    fold = str(args.fold)
    context = args.context
    cuda = args.cuda
    dataset = 'dailydialog'
    doc_stride = args.doc_stride
    overwrite_output_dir = args.overwrite_output_dir
    
    if context == False:
        max_q_length, max_c_length, max_a_length = 400, 400, 160
    elif context == True:
        max_q_length, max_c_length, max_a_length = 512, 512, 200
        
    if context == False:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-qa-without-context-fold' + fold + '/test_predictions.pkl'
        x_train = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_train_without_context.json'))
        x_valid = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_valid_without_context.json'))
        x_test  = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_test_without_context.json'))
    else:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-qa-with-context-fold' + fold + '/test_predictions.pkl'
        x_train = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_train_with_context.json'))
        x_valid = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_valid_with_context.json'))
        x_test  = json.load(open('data/qa/fold' + fold + '/dailydialog_qa_test_with_context.json'))
    

    ###### only train on first 100 samples
    x_train = x_train[:5000]
    x_test = x_test[:1000]
    x_valid = x_valid[:1000]
    ############

    if fold == '1':
        num_steps = int(len(x_train)/batch_size)  # 27915
    else:
        num_steps = int(len(x_train)/batch_size)
    
    train_args = {
        'fp16': False,
        'overwrite_output_dir': True, 
        'doc_stride': doc_stride, 
        'max_query_length': max_q_length, 
        'max_answer_length': max_a_length,
        "max_seq_length": max_c_length,
        'n_best_size': 20,
        'null_score_diff_threshold': 0.0,
        'learning_rate': lr,
        'sliding_window': args.slide,
        'output_dir': save_dir,
        'best_model_dir': save_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': num_steps,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': -1, #500000
        'train_batch_size': batch_size,
        'num_train_epochs': epochs,
        'overwrite_output_dir': overwrite_output_dir,
        'evaluate_during_training': True,
        'wandb_project': "reccon", 
    }


    qa_model = QuestionAnsweringModel(model_family[model], model_exact_id[model], cuda_device=cuda, args=train_args)
    qa_model.train_model(x_train, eval_data=x_valid)
    
    qa_model = QuestionAnsweringModel(model_family[model], save_dir +'best_model/', cuda_device=cuda, args=train_args)
    result, text = qa_model.eval_model(x_test)

    r = evaluate_results(text)
    print (r)
    
    '''
    rf = open('results/dailydialog_qa.txt', 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*40 + '\n')    
    rf.close()
    
    rf = open(result_file, 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*40 + '\n')    
    rf.close()
    
    pickle.dump(text, open(dump_file, 'wb'))
    '''    
