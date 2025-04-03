import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

# class EncoderLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_heads):
#         super(EncoderLayer, self).__init__()
#         self.attn = MultiHeadAttention(input_dim, n_heads)
#         self.AN1 = torch.nn.LayerNorm(input_dim)
#
#         self.l1 = torch.nn.Linear(input_dim, input_dim)
#         self.AN2 = torch.nn.LayerNorm(input_dim)
#
#     def forward(self, X):
#         output = self.attn(X)
#         X = self.AN1(output + X)
#
#         output = self.l1(X)
#         X = self.AN2(output + X)
#
#         return X
#
# class AE1(torch.nn.Module):  # Joining together
#     def __init__(self, vector_size):
#         super(AE1, self).__init__()
#
#         self.vector_size = vector_size
#
#         self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
#         self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)
#
#         self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, num_heads)
#         self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)
#
#         self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
#         self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)
#
#         self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)
#
#         self.dr = torch.nn.Dropout(drop_out_rating)
#
#         self.ac = torch.nn.ReLU()
#
#     def forward(self, X1, X2):
#         X = torch.cat((X1, X2), 1)
#         X = self.dr(self.bn1(self.ac(self.l1(X))))
#
#         X = self.att2(X)
#         X = self.l2(X)
#
#         X_AE = self.dr(self.bn3(self.ac(self.l3(X))))
#
#         X_AE = self.l4(X_AE)
#
#         return X, X_AE
#
# class AE2(torch.nn.Module):
#     def __init__(self, vector_size):
#         super(AE2, self).__init__()
#         self.vector_size = vector_size // 2
#         self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
#         self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)
#         self.att1 = EncoderLayer((self.vector_size + len_after_AE) // 2, num_heads)
#         self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)
#         self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
#         self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)
#         self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)
#         self.dr = torch.nn.Dropout(drop_out_rating)
#         self.ac = torch.nn.ReLU()
#
#     def forward(self, X1, X2):
#         X = X1 + X2
#         X = self.dr(self.bn1(self.ac(self.l1(X))))
#         X = self.att1(X)
#         X = self.l2(X)
#         X_AE = self.dr(self.bn3(self.ac(self.l3(X))))
#         X_AE = self.l4(X_AE)
#         return X, X_AE
#
# class cov(torch.nn.Module):
#     def __init__(self, vector_size):
#         super(cov, self).__init__()
#
#         self.vector_size = vector_size
#         self.co2_1 = torch.nn.Conv2d(1, 1, kernel_size=(2, Cov2Dsize))
#         self.co1_1 = torch.nn.Conv1d(1, 1, kernel_size=Cov1Dsize)
#         self.pool1 = torch.nn.AdaptiveAvgPool1d(len_after_AE)
#         self.ac = torch.nn.ReLU()
#
#     def forward(self, X1, X2):
#         X = torch.cat((X1, X2), 0)
#
#         X = X.view(-1, 1, 2, self.vector_size // 2)
#
#         X = self.ac(self.co2_1(X))
#
#         X = X.view(-1, self.vector_size // 2 - Cov2Dsize + 1, 1)
#         X = X.permute(0, 2, 1)
#         X = self.ac(self.co1_1(X))
#
#         X = self.pool1(X)
#
#         X = X.contiguous().view(-1, len_after_AE)
#
#         return X

class CrossAttention(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, hidden_dim):
        super(CrossAttention, self).__init__()
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        self.hidden_dim = hidden_dim

        # 定义线性变换层，用于计算Query、Key和Value
        self.query_layer = nn.Linear(feature_dim1, hidden_dim)  # 用于第一组特征
        self.key_layer = nn.Linear(feature_dim2, hidden_dim)    # 用于第二组特征
        self.value_layer = nn.Linear(feature_dim2, hidden_dim)  # 用于第二组特征

        self.mlp = nn.Linear(feature_dim1, hidden_dim)
        # 定义softmax层，用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, features1, features2):
        """
        前向传播
        :param features1: 第一组特征，形状为 (batch_size, seq_len1, feature_dim1)
        :param features2: 第二组特征，形状为 (batch_size, seq_len2, feature_dim2)
        :return: 加权后的特征表示，形状为 (batch_size, seq_len1, hidden_dim)
        """
        # 计算Query、Key和Value
        Q = self.query_layer(features1)  # (batch_size, seq_len1, hidden_dim)
        K = self.key_layer(features2)    # (batch_size, seq_len2, hidden_dim)
        V = self.value_layer(features2)  # (batch_size, seq_len2, hidden_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len1, seq_len2)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))  # 缩放
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len1, seq_len2)

        # 计算加权后的Value
        attended_features = torch.matmul(attention_weights, V)  # (batch_size, seq_len1, hidden_dim)
        features1 = self.relu(self.batch_norm1(self.mlp(features1)))
        attended_features = self.batch_norm2(attended_features+features1)

        return attended_features

class BiCrossAttention(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, hidden_dim1, hidden_dim2):
        super(BiCrossAttention, self).__init__()
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # 定义线性变换层，用于计算Query、Key和Value
        self.cross_attention1 = CrossAttention(feature_dim1, feature_dim2, hidden_dim1)
        self.cross_attention2 = CrossAttention(feature_dim2, feature_dim1, hidden_dim1)
        self.attention = MultiHeadAttention(hidden_dim1 * 2, n_heads=2)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim1 * 2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2)
        )
        # 定义softmax层，用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1*2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)
        self.relu = nn.ReLU()

    def forward(self, features1, features2):

        # 计算加权后的Value
        attended_features1 = self.cross_attention1(features1, features2)
        attended_features2 = self.cross_attention2(features2, features1)
        all_attended_features = torch.cat((attended_features1, attended_features2), 1)
        attended_features = self.attention(all_attended_features)
        attended_features = self.relu(self.batch_norm1(attended_features + all_attended_features))

        attended_features_end = self.feed_forward(attended_features)
        # attended_features_end = self.batch_norm2(attended_features_end + attended_features)

        return attended_features_end

# class Interaction(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(Interaction, self).__init__()
#
#         self.ae1 = AE1(input_dim)  # Joining together
#         self.ae2 = AE2(input_dim)
#         # self.cov = cov(input_dim)  # cov
#
#
#     def forward(self, X1, X2):
#         X_aec, X_AE1 = self.ae1(X1, X2)
#         X_aea, X_AE2 = self.ae2(X1, X2)
#         # X_cnn = self.cov(X1, X2)
#
#         # X = torch.cat((X1, X2, X3), 1)
#         # return X_aec, X_aea, X_cnn, X_AE1, X_AE2
#         return X_aec, X_aea, X_AE1, X_AE2

# num_heads = 4
# drop_out_rating = 0.3
# len_after_AE = 128
# Cov1Dsize = 2
# Cov2Dsize = 4

class MFFPDA(nn.Module):
    def __init__(self, probiotics_dim, diseases_dim, embed_dim, batch_size, dropout1, dropout2):
        super(MFFPDA, self).__init__()
        self.probiotics_dim = probiotics_dim
        self.diseases_dim = diseases_dim
        self.batchsize = batch_size
        self.probiotic_dim = self.probiotics_dim // 5
        self.disease_dim = self.diseases_dim //6
        self.total_layer_emb = 768
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2

        self.probiotic_layer = nn.Linear(self.embed_dim*5, self.embed_dim)
        self.probiotic_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer1 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer2 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer3 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer4 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer5 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer5_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer6 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer6_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer7 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer7_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer8 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer8_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.probiotic_layer9 = nn.Linear(self.probiotic_dim, self.embed_dim)
        self.probiotic_layer9_1 = nn.Linear(self.embed_dim, self.embed_dim)


        self.disease_layer = nn.Linear(self.embed_dim*6, self.embed_dim)
        self.disease_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer1 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer2 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer3 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer4 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer5 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer5_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer6 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer6_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer7 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer7_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer8 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer8_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer9 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer9_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.disease_layer10 = nn.Linear(self.disease_dim, self.embed_dim)
        self.disease_layer10_1 = nn.Linear(self.embed_dim, self.embed_dim)



        self.probiotic_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic5_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic6_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic7_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic8_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic9_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.disease_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease5_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease6_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease7_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease8_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease9_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease10_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.probiotic_multihead = MultiHeadAttention(self.embed_dim*5, n_heads=5)
        self.disease_multihead = MultiHeadAttention(self.embed_dim*6, n_heads=6)

        self.attention = BiCrossAttention(self.embed_dim*5, self.embed_dim*6, 512, 128)

        # cnn setting
        self.channel_size = 16
        self.number_map = 5 * 6

        ##外积残差块
        self.Outer_product_rb_1 = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_downsample = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )

        self.Outer_product_conv = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
        )
        self.Outer_product_rb_2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_maxpool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.Outer_product_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.re = nn.ReLU()

        self.Inner_Product_linear = nn.Sequential(
            nn.Linear(self.embed_dim* self.number_map, 1024),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(512, 128)
        )



        self.total_layer = nn.Linear(self.total_layer_emb, self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d((self.channel_size * 4 + 2 * self.embed_dim), momentum=0.5)
        self.con_layer = nn.Sequential(
            nn.Linear(self.channel_size * 4, 512),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(512, 1)
        )
        self.probiotic_fused_network_layer = nn.Linear(self.embed_dim * 5, self.embed_dim)
        self.probiotic_fused_network_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.probiotic_fused_network_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.disease_fused_network_layer = nn.Linear(self.embed_dim * 6, self.embed_dim)
        self.disease_fused_network_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.disease_fused_network_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, probiotic_features, disease_features, device):

        #, probiotic6, probiotic7, probiotic8
        probiotic1, probiotic2, probiotic3, probiotic4, probiotic5 = probiotic_features.chunk(
            5, 1)

        #, disease7, disease8, disease9
        disease1, disease2, disease3, disease4, disease5, disease6 = disease_features.chunk(
            6, 1)

        x_probiotic1 = F.relu(self.probiotic1_bn(self.probiotic_layer1(probiotic1.float().to(device))), inplace=True)
        x_probiotic1 = F.dropout(x_probiotic1, training=self.training, p=self.dropout1)
        x_probiotic1 = self.probiotic_layer1_1(x_probiotic1)

        x_probiotic2 = F.relu(self.probiotic2_bn(self.probiotic_layer2(probiotic2.float().to(device))), inplace=True)
        x_probiotic2 = F.dropout(x_probiotic2, training=self.training, p=self.dropout1)
        x_probiotic2 = self.probiotic_layer2_1(x_probiotic2)

        x_probiotic3 = F.relu(self.probiotic3_bn(self.probiotic_layer3(probiotic3.float().to(device))), inplace=True)
        x_probiotic3 = F.dropout(x_probiotic3, training=self.training, p=self.dropout1)
        x_probiotic3 = self.probiotic_layer3_1(x_probiotic3)

        x_probiotic4 = F.relu(self.probiotic4_bn(self.probiotic_layer4(probiotic4.float().to(device))), inplace=True)
        x_probiotic4 = F.dropout(x_probiotic4, training=self.training, p=self.dropout1)
        x_probiotic4 = self.probiotic_layer4_1(x_probiotic4)

        x_probiotic5 = F.relu(self.probiotic5_bn(self.probiotic_layer5(probiotic5.float().to(device))), inplace=True)
        x_probiotic5 = F.dropout(x_probiotic5, training=self.training, p=self.dropout1)
        x_probiotic5 = self.probiotic_layer5_1(x_probiotic5)

        # x_probiotic6 = F.relu(self.probiotic6_bn(self.probiotic_layer6(probiotic6.float().to(device))), inplace=True)
        # x_probiotic6 = F.dropout(x_probiotic6, training=self.training, p=self.dropout1)
        # x_probiotic6 = self.probiotic_layer6_1(x_probiotic6)
        #
        # x_probiotic7 = F.relu(self.probiotic7_bn(self.probiotic_layer7(probiotic7.float().to(device))), inplace=True)
        # x_probiotic7 = F.dropout(x_probiotic7, training=self.training, p=self.dropout1)
        # x_probiotic7 = self.probiotic_layer7_1(x_probiotic7)
        #
        # x_probiotic8 = F.relu(self.probiotic8_bn(self.probiotic_layer8(probiotic8.float().to(device))), inplace=True)
        # x_probiotic8 = F.dropout(x_probiotic8, training=self.training, p=self.dropout1)
        # x_probiotic8 = self.probiotic_layer8_1(x_probiotic8)

        # x_probiotic9 = F.relu(self.probiotic9_bn(self.probiotic_layer9(probiotic9.float().to(device))), inplace=True)
        # x_probiotic9 = F.dropout(x_probiotic9, training=self.training, p=self.dropout1)
        # x_probiotic9 = self.probiotic_layer9_1(x_probiotic9)

        #, x_probiotic6, x_probiotic7, x_probiotic8
        probiotics = [x_probiotic1, x_probiotic2, x_probiotic3, x_probiotic4, x_probiotic5]

        x_disease1 = F.relu(self.disease1_bn(self.disease_layer1(disease1.float().to(device))), inplace=True)
        x_disease1 = F.dropout(x_disease1, training=self.training, p=self.dropout1)
        x_disease1 = self.disease_layer1_1(x_disease1)

        x_disease2 = F.relu(self.disease2_bn(self.disease_layer2(disease2.float().to(device))), inplace=True)
        x_disease2 = F.dropout(x_disease2, training=self.training, p=self.dropout1)
        x_disease2 = self.disease_layer2_1(x_disease2)

        x_disease3 = F.relu(self.disease3_bn(self.disease_layer3(disease3.float().to(device))), inplace=True)
        x_disease3 = F.dropout(x_disease3, training=self.training, p=self.dropout1)
        x_disease3 = self.disease_layer3_1(x_disease3)

        x_disease4 = F.relu(self.disease4_bn(self.disease_layer4(disease4.float().to(device))), inplace=True)
        x_disease4 = F.dropout(x_disease4, training=self.training, p=self.dropout1)
        x_disease4 = self.disease_layer4_1(x_disease4)

        x_disease5 = F.relu(self.disease5_bn(self.disease_layer5(disease5.float().to(device))), inplace=True)
        x_disease5 = F.dropout(x_disease5, training=self.training, p=self.dropout1)
        x_disease5 = self.disease_layer5_1(x_disease5)

        x_disease6 = F.relu(self.disease6_bn(self.disease_layer6(disease6.float().to(device))), inplace=True)
        x_disease6 = F.dropout(x_disease6, training=self.training, p=self.dropout1)
        x_disease6 = self.disease_layer6_1(x_disease6)

        # x_disease7 = F.relu(self.disease7_bn(self.disease_layer7(disease7.float().to(device))), inplace=True)
        # x_disease7 = F.dropout(x_disease7, training=self.training, p=self.dropout1)
        # x_disease7 = self.disease_layer7_1(x_disease7)
        #
        # x_disease8 = F.relu(self.disease8_bn(self.disease_layer8(disease8.float().to(device))), inplace=True)
        # x_disease8 = F.dropout(x_disease8, training=self.training, p=self.dropout1)
        # x_disease8 = self.disease_layer8_1(x_disease8)
        #
        # x_disease9 = F.relu(self.disease9_bn(self.disease_layer9(disease9.float().to(device))), inplace=True)
        # x_disease9 = F.dropout(x_disease9, training=self.training, p=self.dropout1)
        # x_disease9 = self.disease_layer9_1(x_disease9)

        # x_disease10 = F.relu(self.disease10_bn(self.disease_layer10(disease10.float().to(device))), inplace=True)
        # x_disease10 = F.dropout(x_disease10, training=self.training, p=self.dropout1)
        # x_disease10 = self.disease_layer10_1(x_disease10)

        #, x_disease7, x_disease8, x_disease9
        diseases = [x_disease1, x_disease2, x_disease3, x_disease4, x_disease5, x_disease6]

        all_probiotics = torch.cat(probiotics, dim=1)
        all_diseases = torch.cat(diseases, dim=1)

        probiotic_disease_attention = self.attention(all_probiotics, all_diseases)

        probiotic_fused_network = self.probiotic_multihead(all_probiotics.to(torch.float32))
        disease_fused_network = self.disease_multihead(all_diseases.to(torch.float32))

        probiotic_fused_network = F.relu(self.probiotic_fused_network_bn(
            self.probiotic_fused_network_layer(probiotic_fused_network.float().to(device))),
            inplace=True)
        probiotic_fused_network = F.dropout(probiotic_fused_network, training=self.training, p=self.dropout1)
        probiotic_fused_network = self.probiotic_fused_network_layer_1(probiotic_fused_network)

        disease_fused_network = F.relu(self.disease_fused_network_bn(
            self.disease_fused_network_layer(disease_fused_network.float().to(device))),
            inplace=True)
        disease_fused_network = F.dropout(disease_fused_network, training=self.training, p=self.dropout1)
        disease_fused_network = self.disease_fused_network_layer_1(disease_fused_network)

        #=============外积==========================================
        maps = []
        for i in range(len(probiotics)):
            for j in range(len(diseases)):
                maps.append(torch.bmm(probiotics[i].unsqueeze(2), diseases[j].unsqueeze(1)))
        Outer_product_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            Outer_product_map = torch.cat([Outer_product_map, interaction], dim=1)
        #===============内积============================================
        total = []
        for i in range(len(probiotics)):
            for j in range(len(diseases)):
                total.append(probiotics[i].unsqueeze(1) * diseases[j].unsqueeze(1))

        Inner_Product_map = total[0]
        for i in range(1, len(maps)):
            Inner_Product_map = torch.cat([Inner_Product_map, total[i]], dim=1)
        #====================残差块=====================================================

        Inner_Product = Inner_Product_map.view(Inner_Product_map.shape[0], -1)
        Inner_Product = self.Inner_Product_linear(Inner_Product)

        #########外积残差

        x = self.Outer_product_downsample(Outer_product_map)
        Outer_product_feature_map = self.Outer_product_rb_1(Outer_product_map)
        Outer_product_feature_map = Outer_product_feature_map + x
        Outer_product_feature_map = self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_conv(Outer_product_feature_map)

        x = Outer_product_feature_map
        Outer_product_feature_map = self.Outer_product_rb_2(Outer_product_feature_map)
        Outer_product_feature_map = Outer_product_feature_map + x
        Outer_product_feature_map = self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_maxpool(Outer_product_feature_map)

        Outer_Product = Outer_product_feature_map.view((Inner_Product.shape[0], -1))
        ###########################################################################################################


        ###########################################################################################################
        # Inter = torch.cat((X, Inner_Product, Outer_Product), dim=1)
        Inter = torch.cat((Inner_Product, Outer_Product, probiotic_disease_attention), dim=1)
        # Inter = probiotic_disease_attention
        total = torch.cat((probiotic_fused_network, disease_fused_network, Inter), dim=1)
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)

        regression = self.con_layer(total)
        regression = nn.Sigmoid()(regression)

        return regression.squeeze()
