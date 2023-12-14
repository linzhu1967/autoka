
import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from tqdm.auto import tqdm


KNOWLEDGE_PQAA_DICT = {
    "A": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_A.jsonl",
    "B": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_B.jsonl",
    "C": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_C.jsonl",
    "D": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_D.jsonl",
    "E": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_E.jsonl",
    "F": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_F.jsonl",
    "G": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_G.jsonl",
    "M": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_M.jsonl",
    "N": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_N.jsonl",
    "Z": "/home/datasets/PubMedQA/knowledge/pqaa/teacher_Z.jsonl",
}

KNOWLEDGE_PQAU_DICT = {
    "A": "/home/datasets/PubMedQA/knowledge/pqau/teacher_A.jsonl",
    "B": "/home/datasets/PubMedQA/knowledge/pqau/teacher_B.jsonl",
    "C": "/home/datasets/PubMedQA/knowledge/pqau/teacher_C.jsonl",
    "D": "/home/datasets/PubMedQA/knowledge/pqau/teacher_D.jsonl",
    "E": "/home/datasets/PubMedQA/knowledge/pqau/teacher_E.jsonl",
    "F": "/home/datasets/PubMedQA/knowledge/pqau/teacher_F.jsonl",
    "G": "/home/datasets/PubMedQA/knowledge/pqau/teacher_G.jsonl",
    "M": "/home/datasets/PubMedQA/knowledge/pqau/teacher_M.jsonl",
    "N": "/home/datasets/PubMedQA/knowledge/pqau/teacher_N.jsonl",
    "Z": "/home/datasets/PubMedQA/knowledge/pqau/teacher_Z.jsonl",
}

KNOWLEDGE_PQAL_DEV_DICT = {
    "A": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_A.jsonl",
    "B": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_B.jsonl",
    "C": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_C.jsonl",
    "D": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_D.jsonl",
    "E": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_E.jsonl",
    "F": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_F.jsonl",
    "G": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_G.jsonl",
    "M": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_M.jsonl",
    "N": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_N.jsonl",
    "Z": "/home/datasets/PubMedQA/knowledge/pqal_dev/teacher_Z.jsonl",
}


def get_knowledge_dict(knowledge_path_list):
    knowledge_dict = {}
    for knowledge_path in knowledge_path_list:
        with open(knowledge_path) as reader:
            for _, line in enumerate(tqdm(reader)):
                line_dict = json.loads(line)
                qid = line_dict["id"].strip()
                logits = line_dict["logits"]
                # label = line_dict["label"]
                # last_hidden_states = line_dict["last_hidden_states"]
                if qid not in knowledge_dict.keys():
                    knowledge_dict[qid] = []
                # knowledge_dict[qid].append((logits, label))
                # knowledge_dict[qid].append((logits, last_hidden_states))
                knowledge_dict[qid].append((logits, "none"))

    
    return knowledge_dict


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# TODO
def knowledge_with_max_logit_entropy_uncertainty(knowledge_path_list, save_path):

    label_num=3
    hidden_size=1024
    new_knowledge_list = []
    knowledge_dict = get_knowledge_dict(knowledge_path_list)

    for qid, k_list in knowledge_dict.items():
        q_logits = []
        q_last_hidden_states = []

        for klg in k_list:
            logits = klg[0]
            hidden_state = klg[1]
            q_logits.extend(logits)
            q_last_hidden_states.extend(hidden_state)

        q_logits = np.asarray(q_logits).reshape(-1,label_num)
        q_last_hidden_states = np.asarray(q_last_hidden_states).reshape(-1, hidden_size)

        # 
        q_probabilities = softmax(q_logits)
        q_entropies = -np.sum(q_probabilities * np.log(q_probabilities + 1e-10), axis=1)


        # q_row_max_logit_columns = np.argmax(q_logits, axis=-1) # 
        # max_indexs = np.where(q_row_max_logit_columns==q_label)[0] # 
        # max_q_entropies = q_entropies[max_indexs] # 
        # max_indexs = max_q_entropies.argsort()[:int(len(max_q_entropies)/2)] #
        top_k = int(len(q_entropies)/2)
        # top_k = 2
        max_indexs = q_entropies.argsort()[:top_k] # 

        # 
        max_logits = q_logits[max_indexs,:]
        avg_max_logits = np.average(max_logits, axis=0)
        max_last_hidden_states = q_last_hidden_states[max_indexs,:]
        avg_max_last_hidden_states = np.average(max_last_hidden_states, axis=0)

        q_label = np.argmax(avg_max_logits)

        # max_entropies = np.max(q_entropies)
        # avg_entropies = np.mean(q_entropies)
        min_entropy = np.min(q_entropies)

        new_knowledge_list.append((qid, avg_max_logits, q_label, avg_max_last_hidden_states, min_entropy))

    # save_path = ".josnl"
    print("Start saving")
    with open(save_path, "w") as writer:
        for item in tqdm(new_knowledge_list):
            knowledge_dict = {
                "id": item[0],
                "logits": item[1].tolist(),
                "label": int(item[2]),
                "last_hidden_states": item[3].tolist(),
                "min_entropy": float(item[4]),
            }
            writer.write(json.dumps(knowledge_dict) + "\n")




def avg_knowledge(knowledge_path_list, save_path):

    knowledge_dict = get_knowledge_dict(knowledge_path_list)
    
    label_num=3
    hidden_size=1024
    avg_knowledge_list = []
    for qid, k_list in knowledge_dict.items():
        avg_logits = [0.0] * label_num
        # avg_last_hidden_state = [0.0] * hidden_size

        for klg in k_list:
            avg_logits.extend(klg[0])
            # avg_last_hidden_state.extend(klg[1])
        avg_logits = np.average(np.array(avg_logits).reshape(-1,label_num), axis=0)
        # avg_last_hidden_state = np.average(np.array(avg_last_hidden_state).reshape(-1,hidden_size), axis=0)
        avg_label = np.argmax(avg_logits)
        # avg_knowledge_list.append((qid, avg_logits, avg_label, avg_last_hidden_state))
        avg_knowledge_list.append((qid, avg_logits, avg_label))

    
    # save_path = ".josnl"
    print("Start saving")
    with open(save_path, "w") as writer:
        for item in tqdm(avg_knowledge_list):
            knowledge_dict = {
                "id": item[0],
                "logits": item[1].tolist(),
                "label": int(item[2]),
                # "last_hidden_states": item[3].tolist(),
                # "first_hidden_states": item[4].tolist(),
            }
            writer.write(json.dumps(knowledge_dict) + "\n")


def get_avg_knowledge_by_combination_name(combination_name, pqau_saved_path, pqal_dev_saved_path):

    if os.path.exists(pqau_saved_path) and os.path.exists(pqal_dev_saved_path):
        return pqau_saved_path, pqal_dev_saved_path
    
    teachers_list = list(combination_name)
    knowledge_pqau_path_list = []
    knowledge_pqal_dev_path_list = []
    for t in teachers_list:
        knowledge_pqau_path_list.append(KNOWLEDGE_PQAU_DICT[t])
        knowledge_pqal_dev_path_list.append(KNOWLEDGE_PQAL_DEV_DICT[t])
    
    avg_knowledge(knowledge_pqau_path_list, pqau_saved_path)
    print("Save the averaged pqau knowledge to: {}".format(pqau_saved_path))
    avg_knowledge(knowledge_pqal_dev_path_list, pqal_dev_saved_path)
    print("Save the averaged pqal_dev knowledge to: {}".format(pqal_dev_saved_path))

    return pqau_saved_path, pqal_dev_saved_path


def get_qpaa_avg_knowledge_by_combination_name(combination_name, pqaa_saved_path):
    if os.path.exists(pqaa_saved_path):
        return pqaa_saved_path
    
    teachers_list = list(combination_name)
    knowledge_pqaa_path_list = []
    for t in teachers_list:
        knowledge_pqaa_path_list.append(KNOWLEDGE_PQAA_DICT[t])
    
    avg_knowledge(knowledge_pqaa_path_list, pqaa_saved_path)
    print("Save the averaged pqaa knowledge to: {}".format(pqaa_saved_path))

    # return pqau_saved_path, pqal_dev_saved_path


 
def get_max_knowledge(combination_klg_dir, combination_rep_path, klg_type):
    # step-1: 
    teachers_klg = {}
    if klg_type == "pqau":
        for t_name in KNOWLEDGE_PQAU_DICT.keys():
            teachers_klg[t_name] = get_knowledge_dict([KNOWLEDGE_PQAU_DICT[t_name]])
    elif klg_type == "pqal_dev":
        for t_name in KNOWLEDGE_PQAL_DEV_DICT.keys():
            teachers_klg[t_name] = get_knowledge_dict([KNOWLEDGE_PQAL_DEV_DICT[t_name]])

    # step-2: 
    teacher_name = ['Z','N','M','G','F','E','D','C','B','A']
    for i in range (1, 1024):
        ob_i = '{:010b}'.format(i)
        list_i = np.array([int(ii) for ii in ob_i])
        teacher_idx = np.nonzero(list_i)[0]
        combination_name = "".join(sorted([teacher_name[idx] for idx in teacher_idx]))
        print(i, combination_name, ob_i)
        if len(combination_name) == 1:
            t_idx = teacher_name.index(combination_name)
            list_i[t_idx] = 500
            weighted_i_str = "-".join([str(w) for w in list_i])
            print(weighted_i_str)
            with open(combination_rep_path, "a") as ww:
                ww.write("\t".join([str(i), combination_name, weighted_i_str, ob_i]) + "\n")
            continue
        combination_klg_save_path = combination_klg_dir+"teacher_"+combination_name+".jsonl"
        
        combination_klg = {}
        for t_name in combination_name:
            for qid, klg in teachers_klg[t_name].items():
                # step-3: 
                t_logits = klg[0][0] #
                t_label = np.argmax(np.asarray(t_logits))
                t_max_logit = t_logits[t_label]
                if qid not in combination_klg.keys():
                    combination_klg[qid] = [t_name, t_max_logit, t_logits, t_label]
                else:
                    if t_max_logit > combination_klg[qid][1]:
                        combination_klg[qid] = [t_name, t_max_logit, t_logits, t_label]
        
        # step-4:
        # print("Start saving")
        teacher_percent = {}
        with open(combination_klg_save_path, "w") as writer:
            for qid, klg in combination_klg.items():
                knowledge_dict = {
                    "id": qid,
                    "logits": klg[2],
                    "label": int(klg[3]),
                    "t_name": klg[0],
                }
                writer.write(json.dumps(knowledge_dict) + "\n")

                if klg[0] not in teacher_percent.keys():
                    teacher_percent[klg[0]] = 1
                else:
                    teacher_percent[klg[0]] += 1


        for t_name in sorted(teacher_percent.keys()):
            t_idx = teacher_name.index(t_name)
            list_i[t_idx] = teacher_percent[t_name]
        weighted_i_str = "-".join([str(w) for w in list_i])
        print(weighted_i_str)
        with open(combination_rep_path, "a") as ww:
            ww.write("\t".join([str(i), combination_name, weighted_i_str, ob_i]) + "\n")


def get_max_representations(rep_path_list,rep_save_path):
    rep_dict = {}
    for rep_path in rep_path_list:
        with open(rep_path, "r") as reader:
            for _, line in enumerate(tqdm(reader)):
                idx, combination_name, weighted_i_str, ob_i = line.strip().split("\t")
                if idx == "index":
                    continue
                if idx not in rep_dict.keys():
                    rep_dict[idx] = (combination_name, weighted_i_str, ob_i)
                else:
                    weighted_i = [int(w) for w in weighted_i_str.split("-")]
                    weighted_i_old = [int(w) for w in rep_dict[idx][1].split("-")]
                    all_i = np.array(weighted_i) + np.array(weighted_i_old)
                    all_i = all_i/(500.0+61249.0)
                    all_i_str = "-".join([str(w) for w in all_i])
                    rep_dict[idx] = (combination_name, all_i_str, ob_i)
    
    with open(rep_save_path, "w") as ww:
        ww.write("\t".join(["index", "combination_name", "combination_value", "ob_index"]) + "\n")
        for idx, item in rep_dict.items():
            ww.write("\t".join([str(idx), item[0], item[1], item[2]]) + "\n")
        

# dis-avg   
def get_dis_avg_knowledge_by_combination_name(combination_name, pqau_saved_path, pqal_dev_saved_path):

    if os.path.exists(pqau_saved_path) and os.path.exists(pqal_dev_saved_path):
        return pqau_saved_path, pqal_dev_saved_path
    
    teachers_list = list(combination_name)
    knowledge_pqau_path_list = []
    knowledge_pqal_dev_path_list = []
    for t in teachers_list:
        knowledge_pqau_path_list.append(KNOWLEDGE_PQAU_DICT[t])
        knowledge_pqal_dev_path_list.append(KNOWLEDGE_PQAL_DEV_DICT[t])
    
    dis_avg_knowledge(knowledge_pqau_path_list, pqau_saved_path)
    print("Save the disagreement averaged pqau knowledge to: {}".format(pqau_saved_path))
    dis_avg_knowledge(knowledge_pqal_dev_path_list, pqal_dev_saved_path)
    print("Save the disagreement averaged pqal_dev knowledge to: {}".format(pqal_dev_saved_path))

    return pqau_saved_path, pqal_dev_saved_path


def dis_avg_knowledge(knowledge_path_list, save_path):

    knowledge_dict = get_knowledge_dict(knowledge_path_list)
    
    avg_knowledge_list = []
    for qid, k_list in knowledge_dict.items():
        k_logits = []
        for klg in k_list:
            k_logits.append(klg[0])
        k_len = len(k_logits)

        k_logits = np.array(k_logits)

        avg_logits = np.average(k_logits, axis=0)
        avg_label = np.argmax(avg_logits)
        
        avg_kldiv = 0.0
        for i in range(k_len):
            for j in range(k_len):
                if i != j:
                    kl_div = kl_divergence(k_logits[i], k_logits[j])
                    avg_kldiv += kl_div
        avg_kldiv = avg_kldiv/(k_len*(k_len-1))
        avg_knowledge_list.append((qid, avg_logits, avg_label, avg_kldiv))


    sorted_avg_knowledge_list = sorted(avg_knowledge_list, key=lambda x: x[-1])

    # save_path = ".josnl"
    print("Start saving")
    with open(save_path, "w") as writer:
        for item in tqdm(sorted_avg_knowledge_list):
            knowledge_dict = {
                "id": item[0],
                "logits": item[1].tolist(),
                "label": int(item[2]),
                "kldiv": float(item[3]),
            }
            writer.write(json.dumps(knowledge_dict) + "\n")

def kl_divergence(p, q):
    p = softmax(p)
    q = softmax(q)
    return np.sum(p * np.log(p / q))


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


def concate_knowledge_with_disagreement(knowledge_path_list, save_path, dis_threshold):
    knowledge_dict = get_knowledge_dict(knowledge_path_list)
    concate_knowledge_list = []

    for qid, k_list in knowledge_dict.items():
        k_logits = []
        for klg in k_list:
            k_logits.append(klg[0])

        k_len = len(k_logits)
        k_logits = np.array(k_logits)
        concate_logits = k_logits.flatten() # 
        avg_logits = np.average(k_logits, axis=0)
        avg_label = np.argmax(avg_logits)
        
        avg_kldiv = 0.0
        for i in range(k_len):
            for j in range(k_len):
                if i != j:
                    kl_div = kl_divergence(k_logits[i], k_logits[j])
                    avg_kldiv += kl_div
        avg_kldiv = avg_kldiv/(k_len*(k_len-1))

        if avg_kldiv > dis_threshold:
            alpha = np.array([1, 1, 1])
            random_logits = np.random.dirichlet(alpha)
            avg_label = np.argmax(random_logits)
            concate_logits = np.tile(random_logits, len(knowledge_path_list))  # 

        concate_knowledge_list.append((qid, concate_logits, avg_label, avg_kldiv))

    
    sorted_concate_knowledge_list = sorted(concate_knowledge_list, key=lambda x: x[-1])
 
    # save_path = ".josnl"
    print("Start saving")
    with open(save_path, "w") as writer:
        for item in tqdm(sorted_concate_knowledge_list):
            knowledge_dict = {
                "id": item[0],
                "logits": item[1].tolist(),
                "label": int(item[2]),
                "kldiv": float(item[3]),
            }
            writer.write(json.dumps(knowledge_dict) + "\n")


# ka
def get_concate_logits_with_disagreement(combination_name, pqau_saved_path, pqal_dev_saved_path, dis_threshold_list):
    if os.path.exists(pqau_saved_path) and os.path.exists(pqal_dev_saved_path):
        return pqau_saved_path, pqal_dev_saved_path
    # if len(combination_name) == 1:
    #     return pqau_saved_path, pqal_dev_saved_path

    
    print("start get_concate_logits_with_disagreement")
    teachers_list = list(combination_name)
    knowledge_pqau_path_list = []
    knowledge_pqal_dev_path_list = []
    for t in teachers_list:
        knowledge_pqau_path_list.append(KNOWLEDGE_PQAU_DICT[t])
        knowledge_pqal_dev_path_list.append(KNOWLEDGE_PQAL_DEV_DICT[t])
    
    if isinstance(dis_threshold_list, list):
        print("The dis_threshold_list is a list.")
    else:
        dis_threshold_list = [dis_threshold_list]

    for dis_threshold in dis_threshold_list:
        concate_knowledge_with_disagreement(knowledge_pqau_path_list, pqau_saved_path, dis_threshold)
        print("Save the averaged pqau knowledge to: {}".format(pqau_saved_path))
        concate_knowledge_with_disagreement(knowledge_pqal_dev_path_list, pqal_dev_saved_path, dis_threshold)
        print("Save the averaged pqal_dev knowledge to: {}".format(pqal_dev_saved_path))

    return pqau_saved_path, pqal_dev_saved_path


# ka-ablation study
def get_concate_logits_with_disagreement_ablation_study(knowledge_pqau_path_list, knowledge_pqal_dev_path_list, pqau_saved_path, pqal_dev_saved_path, dis_threshold):
    if os.path.exists(pqau_saved_path) and os.path.exists(pqal_dev_saved_path):
        return pqau_saved_path, pqal_dev_saved_path

    print("start get_concate_logits_with_disagreement_ablation_study")
    
    concate_knowledge_with_disagreement(knowledge_pqau_path_list, pqau_saved_path, dis_threshold)
    print("Save the averaged pqau knowledge to: {}".format(pqau_saved_path))
    concate_knowledge_with_disagreement(knowledge_pqal_dev_path_list, pqal_dev_saved_path, dis_threshold)
    print("Save the averaged pqal_dev knowledge to: {}".format(pqal_dev_saved_path))

    return pqau_saved_path, pqal_dev_saved_path


