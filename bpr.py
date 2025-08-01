
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from recbole.model.abstract_recommender import GeneralRecommender
# from recbole.model.init import xavier_normal_initialization
# from recbole.model.loss import BPRLoss,BPRLossRuction
# from recbole.model.layers import MLPLayers
# from recbole.utils import InputType
# from recbole.model.init import xavier_uniform_initialization
# class SamplerMF(GeneralRecommender):
#     def __init__(self, config, dataset):
#         super(SamplerMF, self).__init__(config, dataset)
#         self.mf_embedding_size = config["embedding_size"]
#         self.mlp_embedding_size = config["embedding_size"]
#         self.user_embeddings = nn.Embedding(self.n_users, self.mf_embedding_size)
#         self.item_embeddings = nn.Embedding(self.n_items, self.mf_embedding_size)
#         self.user_embeddings.weight.data.normal_(0, 0.1)
#         self.item_embeddings.weight.data.normal_(0, 0.1)
#         self.mlp_hidden_size = config['mlp_hidden_size']
#         self.sampler_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
#         # self.fc = nn.Linear(self.mf_embedding_size * 2, 1)
#         # self.mlp_layers = MLPLayers(
#         #     [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
#         # )
#         self.dropout_prob = 0.2 #config['sampler_drop']
#         self.mlp = MLPLayers(
#             [2 * self.mf_embedding_size] + self.mlp_hidden_size, self.dropout_prob
#         )
#         self.apply(xavier_uniform_initialization)

#     # def forward(self, user_ids, item_ids):
#     #     user_embed = self.user_embeddings(user_ids)
#     #     item_embed = self.item_embeddings(item_ids)
#     #     x = torch.cat([user_embed, item_embed], dim=1)
#     #     return self.attn(x).squeeze()
    
#     def forward(self, user_ids, item_ids):
#         user_embed = self.user_embeddings(user_ids)
#         item_embed = self.item_embeddings(item_ids)
#         # x = torch.cat([user_embed, item_embed], dim=1)
#         # mlp_x = self.mlp(x)
#         # output = self.sampler_layer(mlp_x)
#         output = torch.mul(user_embed,item_embed).sum(1)
#         return output#.squeeze(-1)
        

# class BPR(GeneralRecommender):
#     r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

#     input_type = InputType.PAIRWISE

#     def __init__(self, config, dataset):
#         super(BPR, self).__init__(config, dataset)

#         # load parameters info
#         self.embedding_size = config["embedding_size"]
       

#         # define layers and loss
#         self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
#         self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
#         self.loss = BPRLoss()
#         self.sampler_mf = SamplerMF(config, dataset)
#         self.mf_loss = BPRLossRuction()

#         # parameters initialization
#         self.apply(xavier_normal_initialization)

#     def get_user_embedding(self, user):
#         r"""Get a batch of user embedding tensor according to input user's id.

#         Args:
#             user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
#         """
#         return self.user_embedding(user)

#     def get_item_embedding(self, item):
#         r"""Get a batch of item embedding tensor according to input item's id.

#         Args:
#             item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
#         """
#         return self.item_embedding(item)

#     def forward(self, user, item):
        
#         user_e = self.get_user_embedding(user)
#         item_e = self.get_item_embedding(item)
#         return user_e, item_e

#     # def calculate_loss(self, interaction):
#     #     user = interaction[self.USER_ID]
#     #     pos_item = interaction[self.ITEM_ID]
#     #     neg_item = interaction[self.NEG_ITEM_ID]

#     #     user_e, pos_e = self.forward(user, pos_item)
#     #     neg_e = self.get_item_embedding(neg_item)
#     #     pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
#     #         user_e, neg_e
#     #     ).sum(dim=1)
#     #     loss = self.loss(pos_item_score, neg_item_score)
#     #     return loss
#     def calculate_loss(self, interaction):
#         user = interaction[self.USER_ID]
#         pos_item = interaction[self.ITEM_ID]
#         neg_item = interaction[self.NEG_ITEM_ID]

#         mask = interaction['flag'].view(-1,1)
#         original_user = user[mask.squeeze() == 1]
#         original_pos_item = pos_item[mask.squeeze() == 1]
#         original_neg_item = neg_item[mask.squeeze() == 1]

      
        
#         gen_user = user[mask.squeeze() == 0]
#         gen_pos_item = pos_item[mask.squeeze() == 0]
#         gen_neg_item = neg_item[mask.squeeze() == 0]

#         original_user_embedding, original_pos_embeddings = self.forward(original_user,original_pos_item)
       
#         original_neg_embeddings = self.get_item_embedding(original_neg_item)
#         ori_pos_scores = torch.mul(original_user_embedding, original_pos_embeddings).sum(dim=1)
#         ori_neg_scores = torch.mul(original_user_embedding, original_neg_embeddings).sum(dim=1)
        
      
#         loss_ori =  self.loss(ori_pos_scores, ori_neg_scores)

        
#         probs = torch.softmax(self.sampler_mf(gen_user, gen_pos_item),dim=0)
     

#         combined_users = torch.cat([original_user, gen_user])
#         combined_pos_items = torch.cat([original_pos_item, gen_pos_item])
#         combined_neg_items = torch.cat([original_neg_item, gen_neg_item])
        
#         combined_user_embedding,combined_pos_embeddings = self.forward(combined_users,combined_pos_items)
      
#         combined_neg_embeddings = self.get_item_embedding(combined_neg_items)

#         weight = torch.cat([torch.ones(len(original_user)).cuda(),probs])
#         combine_pos_scores = torch.mul(combined_user_embedding, combined_pos_embeddings).sum(dim=1)
#         combine_neg_scores = torch.mul(combined_user_embedding, combined_neg_embeddings).sum(dim=1)
#         loss_com=  self.mf_loss(combine_pos_scores, combine_neg_scores)
#         loss_com_sun = (loss_com*weight).mean()

#         ori_pro_pos = self.sampler_mf(original_user, original_pos_item)
#         ori_pro_neg = self.sampler_mf(original_user, original_neg_item)
#         all_scores = torch.cat([ori_pro_pos,ori_pro_neg])
#         all_labels = torch.cat([torch.ones(len(ori_pro_pos)).cuda(),torch.zeros(len(ori_pro_neg)).cuda()])
       
#         kl_div = F.kl_div(F.log_softmax(all_scores, dim=0), all_labels, reduction='batchmean')
    
        
#         return loss_com_sun,kl_div #self.loss(output, label)
        
#     def predict(self, interaction):
#         user = interaction[self.USER_ID]
#         item = interaction[self.ITEM_ID]
#         user_e, item_e = self.forward(user, item)
#         return torch.mul(user_e, item_e).sum(dim=1)

#     def full_sort_predict(self, interaction):
#         user = interaction[self.USER_ID]
#         user_e = self.get_user_embedding(user)
#         all_item_e = self.item_embedding.weight
#         score = torch.matmul(user_e, all_item_e.transpose(0, 1))
#         return score.view(-1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss,BPRLossRuction
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
from recbole.model.init import xavier_uniform_initialization
class Sampler(GeneralRecommender):
    def __init__(self, config, dataset):
        super(Sampler, self).__init__(config, dataset)
        self.mf_embedding_size = config["embedding_size"]
        self.mlp_embedding_size = config["embedding_size"]
        self.user_embeddings = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_embeddings.weight.data.normal_(0, 0.1)
        self.item_embeddings.weight.data.normal_(0, 0.1)
        self.mlp_hidden_size = [64]
        self.sampler_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        # self.fc = nn.Linear(self.mf_embedding_size * 2, 1)
        self.dropout_prob = 0.8 #config['ori_sampler_drop']
        self.mlp = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        )
       
        # self.user_mlp = MLPLayers(
        #     [2*self.mf_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        # )
        # self.item_mlp = MLPLayers(
        #     [2*self.mf_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        # )
        self.apply(xavier_uniform_initialization)

 
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embeddings(user_ids)
        item_embed = self.item_embeddings(item_ids)
        x = torch.cat([user_embed, item_embed], dim=1)
        mlp_x = self.mlp(x)
        output = self.sampler_layer(mlp_x)
        return output.squeeze(1)





class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
       

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.sampler_mf = Sampler(config, dataset)
        # self.sampler_ori = Sampler(config, dataset)
        # self.sampler_gen = Sampler(config, dataset)
        
        self.mf_loss = BPRLossRuction()

        # parameters initialization
       
        self.mlp_hidden_size = [64]
        self.sampler_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
       
        self.dropout_prob = 0.8 #config['ori_sampler_drop']
        self.mlp = MLPLayers(
            [2*self.embedding_size] + self.mlp_hidden_size, self.dropout_prob
        )
        self.sampler_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.apply(xavier_normal_initialization)


    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

   
    
    def forward(self, user, item):
        
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    # def calculate_loss(self, interaction):
    #     user = interaction[self.USER_ID]
    #     pos_item = interaction[self.ITEM_ID]
    #     neg_item = interaction[self.NEG_ITEM_ID]

    #     user_e, pos_e = self.forward(user, pos_item)
    #     neg_e = self.get_item_embedding(neg_item)
    #     pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
    #         user_e, neg_e
    #     ).sum(dim=1)
    #     loss = self.loss(pos_item_score, neg_item_score)
    #     return loss
    
    def contrastive_align(self,ori, gen, temperature=0.1):
        """
        ori : Tensor [n, d]  原始 sampler 得到的用户表征
        gen : Tensor [n, d]  生成 sampler 得到的用户表征
        return: scalar loss
        """
        # L2 归一化（可选，但通常能提升稳定性）
        ori = F.normalize(ori, dim=-1)
        gen = F.normalize(gen, dim=-1)

        # 计算 n×n 相似度矩阵
        logits = torch.matmul(ori, gen.T) / temperature     # [n, n]

        # 正样本下标就是 0..n-1
        labels = torch.arange(logits.size(0), device=logits.device)

        # InfoNCE: 行方向 + 列方向对称损失
        loss_i = F.cross_entropy(logits, labels)            # ori→gen
        loss_j = F.cross_entropy(logits.T, labels)          # gen→ori
        return (loss_i + loss_j) / 2
    
    
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        mask = interaction['flag'].view(-1,1)
        original_user = user[mask.squeeze() == 1]
        original_pos_item = pos_item[mask.squeeze() == 1]
        original_neg_item = neg_item[mask.squeeze() == 1]

      
        
        gen_user = user[mask.squeeze() == 0]
        gen_pos_item = pos_item[mask.squeeze() == 0]
        gen_neg_item = neg_item[mask.squeeze() == 0]

        original_user_embedding, original_pos_embeddings = self.forward(original_user,original_pos_item)
       
        original_neg_embeddings = self.get_item_embedding(original_neg_item)
        ori_pos_scores = torch.mul(original_user_embedding, original_pos_embeddings).sum(dim=1)
        ori_neg_scores = torch.mul(original_user_embedding, original_neg_embeddings).sum(dim=1)
        
      
        # loss_ori =  self.loss(ori_pos_scores, ori_neg_scores)

        
        # probs = torch.softmax(self.sampler_gen.get_score(gen_user, gen_pos_item),dim=0)
        # probs_ori = torch.softmax(self.sampler_ori.get_score(original_user, original_pos_item),dim=0)
       

        combined_users = torch.cat([original_user, gen_user])
        combined_pos_items = torch.cat([original_pos_item, gen_pos_item])
        combined_neg_items = torch.cat([original_neg_item, gen_neg_item])
        weight = torch.softmax(self.sampler_mf(combined_users,combined_pos_items),dim=0)
        
        combined_user_embedding,combined_pos_embeddings = self.forward(combined_users,combined_pos_items)
        gen_user_embedding,gen_pos_embeddings = self.forward(gen_user,gen_pos_item)
      
        combined_neg_embeddings = self.get_item_embedding(combined_neg_items)

        # weight =torch.cat([probs_ori,probs])
        combine_pos_scores = torch.mul(combined_user_embedding, combined_pos_embeddings).sum(dim=1)
        combine_neg_scores = torch.mul(combined_user_embedding, combined_neg_embeddings).sum(dim=1)
        loss_com=  self.mf_loss(combine_pos_scores, combine_neg_scores)
        loss_com_sun = (loss_com*weight).mean()

        # ori_pro_pos = self.sampler_ori(original_user, original_pos_item)
        # ori_pro_neg = self.sampler_ori(original_user, original_neg_item)
        # # loss_ori =  self.mf_loss(ori_pro_pos, ori_pro_neg).mean()
        # all_scores = torch.cat([ori_pro_pos,ori_pro_neg])
        # all_labels = torch.cat([torch.ones(len(ori_pro_pos)).cuda(),torch.zeros(len(ori_pro_neg)).cuda()])
       
        # kl_div = F.kl_div(F.log_softmax(all_scores, dim=0), all_labels, reduction='batchmean')

        real_logits = self.sampler_mf(original_user,original_pos_item)
        gen_logits = self.sampler_mf(gen_user,gen_pos_item)
        d_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)) + \
           F.binary_cross_entropy_with_logits(
        gen_logits, torch.zeros_like(gen_logits))

        # ori_sampled_item_emb = self.sampler_ori.get_item_embedding(pos_item)
        # gen_sampled_item_emb = self.sampler_gen.get_item_embedding(pos_item)
      
        # cl_loss_user =self.contrastive_align(ori_sampled_emb,gen_sampled_emb)
        # cl_loss_item = self.contrastive_align(ori_sampled_item_emb,gen_sampled_item_emb)
        # cl_loss = cl_loss_user+cl_loss_item
        
        return loss_com_sun,d_loss #self.loss(output, label)
        
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
