import numpy as np
from scipy import stats
import subprocess
import os
import sys

from utils.utils import set_seed
# from use_knowledge import get_avg_knowledge_by_combination_name
# from use_knowledge import get_dis_avg_knowledge_by_combination_name
from use_knowledge_medmcqa import get_concate_logits_with_disagreement_medmcqa

# from amalgamate_ddp_efficient import avg_ka_process
# from amalgamate_ddp_efficient import max_ka_process
# from amalgamate_ddp_efficient import dis_avg_ka_process
# from amalgamate_ddp_efficient import new_ka_process, new_simple_ka_process
from amalgamate_ddp_efficient_medmcqa import new_ka_medmcqa, new_single_ka_medmcqa

import torch
import torch.distributed as dist


def load_combinations_info(rep_path, acc_path):
    combination_name_dict = {}
    combination_weights_dict = {}
    combination_accuracy_dict = {}
    combination_f1_dict = {}

    with open(rep_path, "r") as rep_reader:
        for line in rep_reader:
            line = line.strip().split("\t")
            if line[0] == "index":
                continue 
            idx = int(line[0])
            name = line[1]
            weights = [float(w) for w in line[2].split("-")]
            if idx not in combination_name_dict.keys():
                combination_name_dict[idx] = name
            else:
                print("Duplicated combination with idx{} and name{}".format(idx,name))

            if idx not in combination_weights_dict.keys():
                combination_weights_dict[idx] = weights
            else:
                print("Duplicated combination with idx{} and name{} in combination_weights_dict".format(idx,name))

    with open(acc_path, "r") as acc_reader:
        for line in acc_reader:
            line = line.strip().split("\t")
            if line[0] == "index":
                continue #
            idx = int(line[0])
            acc = float(line[2])
            f1_macro = float(line[3])

            if idx not in combination_accuracy_dict.keys():
                combination_accuracy_dict[idx] = float(acc)
            else:
                print("Duplicated combination with idx{} and name{} in combination_accuracy_dict".format(idx,line[1]))

            if idx not in combination_f1_dict.keys():
                combination_f1_dict[idx] = f1_macro
            else:
                print("Duplicated combination with idx{} and name{} in combination_f1_dict".format(idx,line[1]))

    return combination_name_dict, combination_weights_dict, combination_accuracy_dict, combination_f1_dict



def sample_medmcqa(combination_name, device, random_value):
    if len(combination_name) == 1:
        student_train_knowledge_saved_path="/home/ylz/ad-hoc_datasets/MedMCQA/knowledge2/student_train/teacher_{}.jsonl".format(combination_name)
    else:
        student_train_knowledge_saved_path="/home/ylz/ad-hoc_datasets/MedMCQA/knowledge2_kldiv4.0/student_train/teacher_{}.jsonl".format(combination_name)
    
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    if local_rank in [-1,0]:
        # print("local_rank({}) start get_dis_avg_knowledge_by_combination_name".format(local_rank))
        dis_threshold = 0.0
        get_concate_logits_with_disagreement_medmcqa(combination_name, student_train_knowledge_saved_path, dis_threshold)
    dist.barrier()

    if len(combination_name) == 1:
        # (avg logits)+(psuedo label)
        acc, f1_macro = new_single_ka_medmcqa(combination_name, [student_train_knowledge_saved_path],device, random_value)
    else:
        acc, f1_macro = new_ka_medmcqa(combination_name, [student_train_knowledge_saved_path],device, random_value)
    return [acc, f1_macro]




def gaussian_kernel_fast(x1, x2, l=0.5, sigma_f=0.2):
    dist_matrix = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    
    return sigma_f ** 2 * np.exp(- dist_matrix /(2*(l**2)) )



def gaussian_kernel(x1, x2, l=0.5, sigma_f=0.2):
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m,n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i]-x2[j]) ** 2)
            
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)


def update(X, X_star, Y): # X: np.array # X_star: np.array
    K_YY = gaussian_kernel_fast(X, X)  # K(X,X)
    K_ff = gaussian_kernel_fast(X_star, X_star)  # K(X*,X*)
    K_Yf = gaussian_kernel_fast(X, X_star)  # K(X, X*)
    K_fY = K_Yf.T # K(X*, X) 
    K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X))) 
    
    mu_star = K_fY.dot(K_YY_inv).dot(Y)
    cov_star = K_ff - K_fY.dot(K_YY_inv).dot(K_Yf)
    return mu_star, cov_star    


def cal_EI(mu, cov, best_sample):
    mu = mu.flatten()
    std = np.sqrt(np.diagonal(cov)) # 
    xi = 0.01
    
    Z_ = mu-best_sample-xi
    EI = np.where(std>0, Z_*(1-stats.norm.cdf(Z_/std)) + std*stats.norm.pdf(Z_/std) , 0)
    return EI


def predict_next(sampled_combinations, sampled_combinations_accuracy, combination_weights_dict):
    X = sampled_combinations
    X_star = list(combination_weights_dict.keys()) # 
    Y = [sampled_combinations_accuracy.get(idx) for idx in sampled_combinations] # 
    X_weights = np.array([combination_weights_dict[x] for x in X])
    # print(X_weights.shape)
    X_star_weights = np.array(list(combination_weights_dict.values()))
    # print(X_star_weights.shape)

    mu_star, cov_star = update(X_weights, X_star_weights, Y)
    # Y_star = mu_star.ravel()
    # uncertainty = 1.96 * np.sqrt(np.diag(cov_star))#
    
    ei = cal_EI(mu_star, cov_star, max(Y))
    next_comb_idx = X_star[np.argmax(ei)]
    return next_comb_idx


if __name__ == "__main__":

    seed_value = 9201


    random_seed = set_seed(seed_value)
    print("random seed: {}".format(seed_value))

    # init ddp
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    num_gpus=0
    if torch.cuda.is_available():
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device=torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
            # info
            num_gpus = torch.cuda.device_count()
            print("Let's use", num_gpus, "GPUs!")
            # print(device)
        else:
            print("The distributed setting is disable.")
            sys.exit()
    else:
        print("GPU is not available.")
        sys.exit()

   
    rep_path = "/home/ad-hoc_datasets/MedMCQA/knowledge2_kldiv4.0/combination_representations.tsv"
    acc_path = "/home/ad-hoc_datasets/MedMCQA/knowledge2_kldiv4.0/random9201_combination_accuracy.tsv"
   
    



    combination_weights_dict = {}  
    combination_name_dict = {}      
    sampled_combination_accuracy_dict = {}  # 
    sampled_combination_f1_dict = {}        # 
    # load the info of all combinations
    combination_name_dict, combination_weights_dict, sampled_combination_accuracy_dict, sampled_combination_f1_dict = load_combinations_info(rep_path, acc_path)
    print(sampled_combination_accuracy_dict)
    
    all_combinations_idx = np.array(list(combination_weights_dict.keys()))
    print("length of all combinations: {}".format(len(all_combinations_idx)))
    sampled_labels = np.zeros(len(all_combinations_idx))
    sampled_combinations = np.array(list(sampled_combination_accuracy_dict.keys()))

    

    # init with 10 teacher models and 10 teacher combinations
    init_single_teachers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # init_single_teachers = [16383]
    init_sampled_combinations = np.array(init_single_teachers)
    
    print(init_sampled_combinations) 
    for i, c_idx in enumerate(init_sampled_combinations):
        if c_idx not in sampled_combination_accuracy_dict.keys() or len(init_sampled_combinations) <10:
            c_name = combination_name_dict[c_idx]
            print("The init combination {} {}".format(c_idx, c_name))
            acc, f1_macro = sample_medmcqa(c_name, device, seed_value)
            sampled_combination_accuracy_dict[c_idx] = acc
            sampled_combination_f1_dict[c_idx] = f1_macro
            if local_rank == 1:
                with open(acc_path, "a") as fadd:
                    fadd.write("\n"+"\t".join([str(c_idx), c_name, str(round(acc, 3)), str(round(f1_macro, 3))]))
                fadd.close()

            dist.barrier()
    
    print("length of sampled combinations: {}".format(len(sampled_combinations)))
    
    num_sample = 500
    last_next_combination_index = -1
    for i in range(num_sample):
        next_combination_index = predict_next(sampled_combinations, sampled_combination_accuracy_dict, combination_weights_dict)
        next_combination_name = combination_name_dict[next_combination_index]
        print("The next combination {} {}".format(next_combination_index, next_combination_name))
        
        
        if next_combination_index != last_next_combination_index:
            last_next_combination_index = next_combination_index
        else:
            print("The next combination {} {} is the same as last time!".format(next_combination_index, next_combination_name))
            break
        
        if next_combination_index in sampled_combination_accuracy_dict:
            acc = sampled_combination_accuracy_dict[next_combination_index]
            f1_macro = 0.000
            
        else:
            acc, f1_macro = sample_medmcqa(next_combination_name, device, seed_value)
            sampled_combination_accuracy_dict[next_combination_index] = acc
            sampled_combination_f1_dict[next_combination_index] = f1_macro
            if local_rank == 1:
                with open(acc_path, "a") as fadd:
                    fadd.write("\n"+"\t".join([str(next_combination_index), next_combination_name, str(round(acc, 3)), str(round(f1_macro, 3))]))
                fadd.close()
        dist.barrier()   
        sampled_combinations = np.append(sampled_combinations, next_combination_index)
    
    # print(sampled_combination_accuracy_dict)
    if local_rank == 1:
        for k in sampled_combination_accuracy_dict:
            print("\t".join([str(k), combination_name_dict[k], str(sampled_combination_accuracy_dict[k]), str(sampled_combination_f1_dict[k])]))



