import numpy as np
import pandas as pd
from io import StringIO
import pickle
import random
#random.seed(42)

def get_prompt_conclass(inital_prompt, numbering, n_samples_per_class,nclass,nset, name_cols):
    prompt=""
    for i in range(nset):
        prompt+=name_cols
        for j in range(10): 
            prompt+=f'{numbering[j]}.\n'
            for k in range(n_samples_per_class):
                prompt +='{'+f'v{i*(n_samples_per_class*10)+j*n_samples_per_class+k}'+'}'
            prompt += f'\n'
        prompt += f'\n'      
    prompt+=name_cols 
    prompt = inital_prompt+prompt
    return prompt
    
def filtering_categorical(result_df, categorical_features, unique_features):
    org_df = result_df.copy()
    shape_before = org_df.shape
    Target = 'user_id' #'user_id'
    for column in categorical_features:
        if column==Target:
            print(result_df)
            try:
                result_df = result_df[result_df[column].map(lambda x: int(x) in unique_features[column])]
            except Exception as e:
                print(f"Error occurred: {e}")
                result_df = None
        else:
            try:
                result_df = result_df[result_df[column].map(lambda x: int(x) in unique_features[column])]
            except Exception as e:
                print(f"Error occurred: {e}")
                result_df = None
           
    # if shape_before!=result_df.shape:
    #     for column in categorical_features:
    #         filtered = org_df[org_df[column].map(lambda x: x not in unique_features[column])]
    return result_df
    
def parse_prompt2df(one_prompt, split, inital_prompt, col_name):
    one_prompt = one_prompt.replace(inital_prompt, '')
    input_prompt_data = one_prompt.split(split)
    input_prompt_data = [x for x in input_prompt_data if x]
    input_prompt_data = '\n'.join(input_prompt_data)
    input_df = pd.read_csv(StringIO(input_prompt_data), sep=",", header=None, names=col_name)
    input_df = input_df.dropna()
    return input_df


def parse_result(one_prompt, name_cols, col_name, categorical_features, unique_features, filter_flag=True):
    one_prompt = one_prompt.replace(name_cols, '')
 
    try:
        result_df = pd.read_csv(StringIO(one_prompt), sep=",", header=None, names=col_name, on_bad_lines="skip")# names=col_name,
        result_df = result_df.dropna()
        if filter_flag:
            result_df = filtering_categorical(result_df, categorical_features, unique_features) 
    
    except Exception as e:
        # 如果列数不匹配，跳过当前行
        result_df = None
 
  
  
   
    return result_df
    

def get_unique_features(data, categorical_features):
    unique_features={}
    for column in categorical_features:
        try:
            unique_features[column] = sorted(data[column].unique())
        except:
            unique_features[column] = data[column].unique()
    return unique_features


def get_sampleidx_from_data(unique_features, target, n_samples_total, n_batch, n_samples_per_class, nset, name_cols, data):
    # input sampling
    unique_classes = unique_features[target]
    random_idx_batch_list=[]
    target_df_list=[]
    for c in unique_classes:
        target_df=data[data[target]==c]
        if len(target_df) < n_samples_total:
            replace_flag=True
        else:
            replace_flag=False
        random_idx_batch = np.random.choice(len(target_df), n_samples_total, replace=replace_flag)
        random_idx_batch = random_idx_batch.reshape(n_batch,nset,1,n_samples_per_class)
        random_idx_batch_list.append(random_idx_batch)
        target_df_list.append(target_df)
    random_idx_batch_list = np.concatenate(random_idx_batch_list, axis=2)
    return random_idx_batch_list, target_df_list


# def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass):
#     fv_cols = ('{},' * len(data.columns))[:-1] + '\n'
#     # 初始化输入批次列表
#     inputs_batch = []
#     batch_size =32
#     n_class_batch = (nclass + batch_size - 1) // batch_size
#     ids = list(range(nclass))
#     # 遍历类别批次    
#     for batch_idx in range(n_batch):
        
#         random.shuffle(ids)
#         batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        
#         for class_batch_idx in range(n_class_batch):
#             class_start = class_batch_idx * batch_size
#             class_end = min(class_start + batch_size, nclass)
#             # 使用向量化操作批量处理数据
#             inputs = {}
#             for i in range(nset):
#                 # 获取当前 set 和 batch 的随机索引
#                 # 获取对应的目标 DataFrame
#                 if (class_end-class_start)<batch_size:
#                     gap = batch_size-(class_end-class_start)
#                     target_ins = [target_df_list[j] for j in batches[class_batch_idx]]  #range(class_start, class_end)
#                     target_lst = [target_df_list[class_end-1]]*gap
#                     target_dfs =  target_ins+ target_lst  
#                     idx_batch1 = random_idx_batch_list[batch_idx, i, batches[class_batch_idx], :]  
#                     idx_batch2 = np.array((list(random_idx_batch_list[batch_idx, i, class_end-1, :].reshape(-1,n_samples_per_class)))*gap)
#                     idx_batch =  np.vstack((idx_batch1, idx_batch2))
                    
#                 else:  
                       
#                     target_dfs = [target_df_list[j] for j in batches[class_batch_idx]]
#                     idx_batch = random_idx_batch_list[batch_idx, i, batches[class_batch_idx], :]
                  
                
#                     # 使用 NumPy 的向量化操作批量获取数据
#                 for j, target_df in enumerate(target_dfs):
#                     # 获取当前类别的样本索引
#                     idx = idx_batch[j]
                    
#                     # 使用 Pandas 的 iloc 批量获取数据
#                     samples = target_df.iloc[idx].values
                    
#                     # 格式化数据并存储到 inputs 字典中
#                     for k, sample in enumerate(samples):
#                         key = f'v{i * (n_samples_per_class * batch_size) + j * n_samples_per_class + k}'
#                         inputs[key] = fv_cols.format(*sample)
                      
                
#             inputs_batch.append(inputs)
          
    
#     return inputs_batch

# def get_relations():
#     file = 'dataset/ml/step1/befor_syn/top_5_related_users.pkl'
#     # 读取.pkl文件
#     with open(file, 'rb') as f:
#         user_relations = pickle.load(f)
    
#     # 返回每个用户的朋友用户列表
#     return user_relations
# def expand_batches_with_friends(batches, user_relations):
#     """
#     对于batches中的每个用户，获取其5个朋友用户，并将这些朋友用户添加到原列表中。
    
#     :param batches: list of lists，每个子列表包含5个用户ID
#     :param user_relations: dict，键是用户ID，值是长度为5的列表，表示该用户的朋友用户
#     :return: list，每个子列表扩展为长度为30的列表
#     """
#     expanded_batches = []
    
#     for batch in batches:
#         expanded_batch = []
#         for user in batch:
#             # 获取该用户的朋友用户
#             userid = user+1
#             friends = user_relations.get(userid, [userid] * 5)  # 如果没有找到朋友，用用户自己的ID填充
#             friends_id =[(fid-1) for fid in friends]
#             expanded_batch.extend([user] + friends_id)  # 添加用户及其朋友
#         expanded_batches.append(expanded_batch)
    
#     return expanded_batches

def process_clustered_users( separator='---'):
    """
    处理聚类后的用户数据，将相同簇的用户相邻，不同簇的用户之间用标记隔开。
    
    :param file_path: 聚类后的用户数据文件路径
    :param separator: 不同簇用户之间的分隔标记，默认为 '---'
    :return: 一个包含用户和分隔标记的列表
    """
    # 读取聚类后的用户数据
    clustered_data = pd.read_csv('./dataset/ml/step1/befor_syn/user_clusters_normalized.csv')
    
    # 按cluster排序
    clustered_data = clustered_data.sort_values(by='cluster')
    
    # 初始化结果列表
    result_list = []
    
    # 遍历每个cluster
    for cluster in clustered_data['cluster'].unique():
        # 获取当前cluster的所有用户
        users_in_cluster = clustered_data[clustered_data['cluster'] == cluster]['user_id'].tolist()
        # 将当前cluster的用户添加到结果列表
        result_list.extend(users_in_cluster)
        # 在不同cluster之间添加分隔标记
        result_list.append(separator)
    
    return result_list

def group_users_by_10(user_list, separator='---'):
    """
    将用户列表分组，每组10个用户。如果遇到分隔标记，并且当前组不满10个用户，则用最后一个元素补全。
    首先根据分隔标记将用户分成几个单独的子列表，然后对每个子列表进行分组。
    
    :param user_list: 包含用户和分隔标记的列表
    :param group_size: 每组的用户数量，默认为10
    :param separator: 分隔标记，默认为 '---'
    :return: 一个包含分组用户列表的列表
    """
    # 首先根据分隔标记将用户分成几个单独的子列表
    sublists = []
    current_sublist = []
    
    for user in user_list:
        if user == separator:
            if current_sublist:
                sublists.append(current_sublist)
                current_sublist = []
        else:
            current_sublist.append(user)
    
    # 如果最后一个子列表不为空，添加到子列表列表中
    if current_sublist:
        sublists.append(current_sublist)
    
    # 对每个子列表，每10个用户划分为一组，不满10个用户则用最后一个元素填充
    
    return sublists
    
def spilt_user_list(sublists,group_size=10):
    
    all_grouped_users = []
    
    for sublist in sublists:
        grouped_users = []
        for i in range(0, len(sublist), group_size):
            group = sublist[i:i + group_size]
            # 如果当前组不满10个用户，用最后一个元素补全
            while len(group) < group_size:
                group.append(group[-1])
            grouped_users.append(group)
        all_grouped_users.extend(grouped_users)
    
    return all_grouped_users


# def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass):
#     fv_cols = ('{},' * len(data.columns))[:-1] + '\n'
#     # 初始化输入批次列表
#     inputs_batch = []
#     # batch_users = 5
#     # batch_relations = 5
#     # batch_size = 5
#     # batch_total_numbers = batch_users+batch_users*batch_relations
#     # user_relations = get_relations()
#     result_list = process_clustered_users()
#     sub_list = group_users_by_10(result_list)
#     grouped_users_list = spilt_user_list(sub_list)
#     n_class_batch = len(grouped_users_list)
#     ids = list(range(nclass))
#     # 遍历类别批次    
#     for batch_idx in range(n_batch):
        
#         random.shuffle(ids)
#         batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
#         expanded_batches = expand_batches_with_friends(batches,user_relations)
        
#         for class_batch_idx in range(n_class_batch):
#             class_start = class_batch_idx * batch_size
#             class_end = min(class_start + batch_size, nclass)
#             # 使用向量化操作批量处理数据
#             inputs = {}
#             for i in range(nset):
#                 # 获取当前 set 和 batch 的随机索引
#                 # 获取对应的目标 DataFrame
#                 if (class_end-class_start)<batch_size:
#                     gap = batch_size-(class_end-class_start)
#                     target_ins = [target_df_list[j] for j in expanded_batches[class_batch_idx]]  #range(class_start, class_end)
#                     target_lst = [target_df_list[class_end-1]]*(gap+gap*batch_relations)
#                     target_dfs =  target_ins+ target_lst  
#                     idx_batch1 = random_idx_batch_list[batch_idx, i, expanded_batches[class_batch_idx], :]  
#                     idx_batch2 = np.array((list(random_idx_batch_list[batch_idx, i, class_end-1, :].reshape(-1,n_samples_per_class)))*(gap+gap*batch_relations))
#                     idx_batch =  np.vstack((idx_batch1, idx_batch2))
                    
#                 else:  
                       
#                     target_dfs = [target_df_list[j] for j in expanded_batches[class_batch_idx]]
#                     idx_batch = random_idx_batch_list[batch_idx, i, expanded_batches[class_batch_idx], :]
                  
                
#                     # 使用 NumPy 的向量化操作批量获取数据
#                 for j, target_df in enumerate(target_dfs):
#                     # 获取当前类别的样本索引
#                     idx = idx_batch[j]
                    
#                     # 使用 Pandas 的 iloc 批量获取数据
#                     samples = target_df.iloc[idx].values
                    
#                     # 格式化数据并存储到 inputs 字典中
#                     for k, sample in enumerate(samples):
#                         key = f'v{i * (n_samples_per_class * 30) + j * n_samples_per_class + k}'
#                         inputs[key] = fv_cols.format(*sample)
                      
                
#             inputs_batch.append(inputs)
          
    
#     return inputs_batch
def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass):
    fv_cols = ('{},' * len(data.columns))[:-1] + '\n'
    # 初始化输入批次列表
    inputs_batch = []
    # batch_users = 5
    # batch_relations = 5
    # batch_size = 5
    # batch_total_numbers = batch_users+batch_users*batch_relations
    # user_relations = get_relations()
    result_list = process_clustered_users()
    sub_list = group_users_by_10(result_list)
    grouped_users_list = spilt_user_list(sub_list)
    n_class_batch = len(grouped_users_list)
  
    # 遍历类别批次    
    for batch_idx in range(n_batch):
        
        for sublist in sub_list:
            random.shuffle(sublist)
        grouped_batches = spilt_user_list(sub_list)
        # random.shuffle(ids)
        # batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        # expanded_batches = expand_batches_with_friends(batches,user_relations)
        
        for class_batch_idx in range(n_class_batch):
            # class_start = class_batch_idx * batch_size
            # class_end = min(class_start + batch_size, nclass)
            # 使用向量化操作批量处理数据
            inputs = {}
            for i in range(nset):
                # 获取当前 set 和 batch 的随机索引
                # 获取对应的目标 DataFrame
               
                batch_group_list = [x - 1 for x in grouped_batches[class_batch_idx]]
                target_dfs = [target_df_list[j] for j in batch_group_list]
                idx_batch = random_idx_batch_list[batch_idx, i, batch_group_list, :]
                  
                
                    # 使用 NumPy 的向量化操作批量获取数据
                for j, target_df in enumerate(target_dfs):
                    # 获取当前类别的样本索引
                    idx = idx_batch[j]
                    
                    # 使用 Pandas 的 iloc 批量获取数据
                    samples = target_df.iloc[idx].values
                    
                    # 格式化数据并存储到 inputs 字典中
                    for k, sample in enumerate(samples):
                        key = f'v{i * (n_samples_per_class * 10) + j * n_samples_per_class + k}'
                        inputs[key] = fv_cols.format(*sample)
                      
                
            inputs_batch.append(inputs)
          
    
    return inputs_batch

    
def make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                      N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS):
    
    random_idx_batch_list, target_df_list = get_sampleidx_from_data(unique_categorical_features, TARGET, 
                                                                    N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, data)
    inputs_batch = get_input_from_idx(target_df_list, random_idx_batch_list, data, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, N_CLASS)
    final_prompt = template1_prompt.batch(inputs_batch)
    return final_prompt, inputs_batch

def useThis(one_prompt):
    char = one_prompt[0]
    if char.isdigit() and int(char) in [0,1,2,3,4]:
        return True, int(char)
    else:
        return False, None
