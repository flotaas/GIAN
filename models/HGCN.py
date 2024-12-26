import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.subNets.hygraph import HypergraphConv, Graph_Attention_Union
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_seq_len=1024, learnable=False):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        else:
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            #pe[:, 1::2] = torch.cos(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
            pe = pe.unsqueeze(0) # Note: pe with size (1, seq_len, feature_dim)
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: with size (batch_size, seq_len, feature_dim)
        :return:
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HyperGCN(nn.Module):
    def __init__(self, args):
        super(HyperGCN, self).__init__()

        self.args = args
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        args.fusion_dim = args.fus_d_l + args.fus_d_a + args.fus_d_v
        orig_d_l, orig_d_a, orig_d_v = args.feature_dims

        self.proj_l = nn.Linear(orig_d_l, args.fus_d_l)
        self.pos_l = PositionalEmbedding(args.fus_d_l)
        self.proj_a = nn.Linear(orig_d_a, args.fus_d_a)
        self.pos_a = PositionalEmbedding(args.fus_d_a)
        self.proj_v = nn.Linear(orig_d_v, args.fus_d_v)
        self.pos_v = PositionalEmbedding(args.fus_d_v)

        self.hypergraph_a = HypergraphConv(in_features=args.fus_d_a, out_features=args.fus_d_a, edges=64, filters=128)
        self.hypergraph_v = HypergraphConv(in_features=self.args.fus_d_v, out_features=args.fus_d_v, edges=64, filters=128)
        self.hypergraph_l = HypergraphConv(in_features=self.args.fus_d_l, out_features=args.fus_d_l, edges=64, filters=128)
        # args.edges=64 for mosi


    def forward(self, text_x, audio_x, video_x):

        x_l = self.text_model(text_x)
        proj_x_l = self.pos_l(self.proj_l(x_l)).transpose(1, 2).unsqueeze(-1)
        proj_x_a = self.pos_a(self.proj_a(audio_x)).transpose(1, 2).unsqueeze(-1)  # batch_size, da, seq_len
        proj_x_v = self.pos_v(self.proj_v(video_x)).transpose(1, 2).unsqueeze(-1)  # batch_size, da, seq_len

        out_a = self.hypergraph_a(proj_x_a)
        out_v = self.hypergraph_v(proj_x_v)
        out_l = self.hypergraph_l(proj_x_l)

        return out_l.squeeze(), out_a.squeeze(), out_v.squeeze()


class MGAT(nn.Module):
    def __init__(self, args):
        super(MGAT, self).__init__()
        self.args = args
        self.gatw = 1
        self.gat_va = Graph_Attention_Union(self.args.fus_d_a, self.args.fus_d_v, meanw=1.3)
        self.gat_vl = Graph_Attention_Union(self.args.fus_d_l, self.args.fus_d_v, meanw=1.3)
        self.gat_av = Graph_Attention_Union(self.args.fus_d_v, self.args.fus_d_a, meanw=1.3)
        self.gat_al = Graph_Attention_Union(self.args.fus_d_l, self.args.fus_d_a, meanw=1.3)
        self.gat_la = Graph_Attention_Union(self.args.fus_d_a, self.args.fus_d_l, meanw=1.3)
        self.gat_lv = Graph_Attention_Union(self.args.fus_d_v, self.args.fus_d_l, meanw=1.3)

    def forward(self, proj_x_l, proj_x_a, proj_x_v):
        # (B, D, T) --> (B, D, T, 1)
        proj_x_l, proj_x_a, proj_x_v = proj_x_l.unsqueeze(-1), proj_x_a.unsqueeze(-1), proj_x_v.unsqueeze(-1)
        # CFL
        proj_v_all = self.gatw * self.gat_av(proj_x_a, proj_x_v) + self.gatw * self.gat_lv(proj_x_l, proj_x_v) + proj_x_v
        proj_a_all = self.gatw * self.gat_va(proj_x_v, proj_x_a) + self.gatw * self.gat_la(proj_x_l, proj_x_a) + proj_x_a
        proj_l_all = self.gatw * self.gat_al(proj_x_a, proj_x_l) + self.gatw * self.gat_vl(proj_x_v, proj_x_l) + proj_x_l

        return proj_l_all.squeeze(), proj_a_all.squeeze(), proj_v_all.squeeze()


class transformer_fusion(nn.Module):
    def __init__(self, args):
        super(transformer_fusion, self).__init__()
        self.fusion_trans = TransformerEncoder(embed_dim=args.fus_d_l + args.fus_d_a + args.fus_d_v,
                                               num_heads=args.fus_nheads, layers=args.fus_layers,
                                               attn_dropout=args.fus_attn_dropout, relu_dropout=args.fus_relu_dropout,
                                               res_dropout=args.fus_res_dropout,
                                               embed_dropout=args.fus_embed_dropout, attn_mask=args.fus_attn_mask)

    def forward(self, text_x, audio_x, video_x):
        # (B, D, T) -->  (T, B, D)
        text_x, audio_x, video_x = text_x.permute(2, 0, 1), audio_x.permute(2, 0, 1), video_x.permute(2, 0, 1)
        trans_seq = self.fusion_trans(torch.cat((text_x, audio_x, video_x), dim=2))
        if type(trans_seq) == tuple:
            trans_seq = trans_seq[0]

        return trans_seq[0]  # Utilize the [CLS] of text for full sequences representation.



class classifier(nn.Module):

    def __init__(self, args):
        super(classifier, self).__init__()
        self.norm = nn.BatchNorm1d(args.fusion_dim)
        self.drop = nn.Dropout(args.clf_dropout)
        self.linear_1 = nn.Linear(args.fusion_dim, args.clf_hidden_dim)
        self.linear_2 = nn.Linear(args.clf_hidden_dim, 1)
        # self.linear_3 = nn.Linear(hidden_size, hidden_size)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, fusion_feature):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(fusion_feature)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = torch.sigmoid(self.linear_2(y_1))
        # 强制将输出结果转化为 [-3,3] 之间
        output = y_2 * self.output_range + self.output_shift
        return output


class HGCN(nn.Module):
    def __init__(self, args):
        super(HGCN, self).__init__()

        self.args = args
        self.TGCN = HyperGCN(args)
        self.MGCN = MGAT(args)
        self.fusion = transformer_fusion(args)
        self.classifier = classifier(args)

    def forward(self, text_x, audio_x, video_x):
        pass
