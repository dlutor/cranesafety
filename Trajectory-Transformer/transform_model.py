# encoding:utf-8
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, d=-2):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            # >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.size(-2)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=.0):
        super().__init__()  # B x S x F
        self.att = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, dropout=dropout)

    def forward(self, q, k, v):
        q = q.transpose(0, 1).contiguous()  ####### s x b x f
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        x, att = self.att(q, k, v)
        x = x.transpose(0, 1).contiguous()  ###### b x s x f
        att = att.transpose(0, 1).contiguous()
        return x, att


class Att_encoder(nn.Module):
    def __init__(self, dim=2, headers=1, feed_dim=10, dropout=0.):
        super(Att_encoder, self).__init__()
        self.self_att = MultiHeadAttention(hid_dim=dim, n_heads=headers, dropout=dropout)
        self.att = MultiHeadAttention(hid_dim=dim, n_heads=headers, dropout=dropout)
        self.feed = nn.Sequential(
            nn.Linear(dim, feed_dim),
            # nn.BatchNorm1d(feed_dim),
            nn.LayerNorm(feed_dim),
            nn.ReLU(),
            #
            nn.Dropout(dropout),
            nn.Linear(feed_dim, dim),
            # nn.ReLU(),
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.layernorm3 = nn.LayerNorm(dim)

    def forward(self, x, m):
        x = self.layernorm1(x)
        x = x + self.self_att(x, x, x)[0]
        x = self.layernorm2(x)
        x = x + self.att(x, m, m)[0]
        x = self.layernorm3(x)
        x = x + self.feed(x)
        return x


class Attention_encoder(nn.Module):
    def __init__(self, output_size=2, output_len=20, hidden_size=2, heads=1, attention_n=1, dropout=0., ):
        super().__init__()
        self.dropout = dropout
        self.muti_atts = nn.ModuleList(
            [Att_encoder(output_len, heads, hidden_size, dropout) for i in range(attention_n)])
        self.pos = PositionalEncoding(output_size, dropout)

    def forward(self, x):
        x = self.pos(x)
        x = x.transpose(-1, -2).contiguous()
        for att in self.muti_atts:
            x = att(x, x)
        x = x.transpose(-1, -2).contiguous()
        return x


class Attention_encoder_p(nn.Module):
    def __init__(self, output_size=2, output_len=20, hidden_size=2, heads=1, attention_n=1, dropout=0., ):
        super().__init__()
        self.dropout = dropout
        self.muti_atts = nn.ModuleList(
            [Att_encoder(output_len, heads, hidden_size, dropout) for i in range(attention_n)])
        self.pos = PositionalEncoding(output_len, dropout)

    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        x = self.pos(x)
        for att in self.muti_atts:
            x = att(x, x)
        x = x.transpose(-1, -2).contiguous()
        return x


class Att_decoder_f(nn.Module):
    def __init__(self, dim=2, headers=1, feed_dim=10, dropout=0.):
        super().__init__()
        self.self_att = MultiHeadAttention(hid_dim=dim, n_heads=headers, dropout=dropout)
        self.att = MultiHeadAttention(hid_dim=dim, n_heads=headers, dropout=dropout)
        self.feed = nn.Sequential(
            nn.Linear(dim, feed_dim),
            nn.LayerNorm(feed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_dim, dim),
            # nn.ReLU(),
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.layernorm3 = nn.LayerNorm(dim)

    def forward(self, x, m):
        b, k, s, f = x.size()
        x = x.view(-1, s, f)
        x = self.layernorm1(x)
        x = x + self.self_att(x, x, x)[0]
        x = self.layernorm2(x)
        x = x + self.att(x, m, m)[0]
        x = self.layernorm3(x)
        x = x + self.feed(x)
        x = x.view(b, k, s, f)
        return x


class Att_decoder_b(nn.Module):
    def __init__(self, dim=2, headers=1, feed_dim=10, dropout=0.):
        super().__init__()
        self.self_att = MultiHeadAttention(dim, headers, dropout)
        self.feed = nn.Sequential(
            nn.Linear(dim, feed_dim),
            nn.ReLU(),
            nn.LayerNorm(feed_dim),
            nn.Dropout(dropout),
            nn.Linear(feed_dim, dim),
            nn.ReLU(),
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, k, s, f = x.size()
        x = self.layernorm1(x)
        x = x.view(-1, s, f)
        x = x + self.self_att(x, x, x)[0]  # .view(b,k,s,f)
        x = x.view(b, k, s, f)
        x = self.layernorm2(x)
        # y = x.view(b, k, -1)
        x = x + self.feed(x).view_as(x)
        return x


class Attention_decoder(nn.Module):
    def __init__(self, output_size=2, output_len=20, hidden_size=2, heads=1, attention_n=1, attention_n1=1,out = 12,
                 dropout=0., ):
        super().__init__()
        self.dropout = dropout
        self.muti_atts = nn.ModuleList(
            [Att_decoder_f(output_len, heads, hidden_size, dropout) for i in range(attention_n)])
        self.muti_atts1 = nn.ModuleList(
            [Att_decoder_b(output_len, heads, hidden_size, dropout) for i in range(attention_n1)])
        self.pos = PositionalEncoding(output_size, dropout)
        self.l = nn.Linear(output_size * output_len, output_size)
        # self.l = nn.Linear(output_len, 1)
        # self.bn = nn.BatchNorm2d(out)

    def forward(self, x, m, e):
        x = self.pos(x)
        b, k, s, f = x.size()
        _, ms, mf = m.size()
        m = m.unsqueeze(-3).repeat((1, k, 1, 1)).view(-1, ms, mf)
        # x = x.view(-1,f,s)
        x = e(x.transpose(-1, -2).contiguous())
        # x = x.transpose(-1, -2).contiguous()
        m = m.transpose(-1, -2).contiguous()
        for att in self.muti_atts:
            x = att(x, m)
        for att in self.muti_atts1:
            x = att(x)
        x = x.transpose(-1, -2).contiguous()
        # x = self.bn(x)
        x = self.l(x.view(b, k, -1))  # .view(b,k,-1)
        # x = self.l(x).view(b,k,-1)
        return x


class Attention_decoder_p(nn.Module):
    def __init__(self, output_size=2, output_len=20, hidden_size=2, heads=1, attention_n=1, attention_n1=1,
                 dropout=0., ):
        super().__init__()
        self.dropout = dropout
        self.muti_atts = nn.ModuleList(
            [Att_decoder_f(output_len, heads, hidden_size, dropout) for i in range(attention_n)])
        self.muti_atts1 = nn.ModuleList(
            [Att_decoder_b(output_len, heads, hidden_size, dropout) for i in range(attention_n1)])
        self.pos = PositionalEncoding(output_len, dropout)
        self.l = nn.Linear(output_size * output_len, output_size)

    def forward(self, x, m):
        b, k, s, f = x.size()
        m = m.unsqueeze(-3).repeat((1, k, 1, 1)).view(-1, s, f)
        # x = x.view(-1,f,s)
        x = x.transpose(-1, -2).contiguous()
        m = m.transpose(-1, -2).contiguous()
        x = self.pos(x)
        for att in self.muti_atts:
            x = att(x, m)
        for att in self.muti_atts1:
            x = att(x)
        x = x.transpose(-1, -2).contiguous()
        x = self.l(x.view(b, k, -1))  # .view(b,k,-1)
        return x

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        w = torch.randn(inp_size,d_model)
        # self.lut = nn.Linear(inp_size, d_model,bias=False)
        self.d_model = d_model
        self.register_buffer('w',w)

    def forward(self, x):
        return x @ self.w * math.sqrt(self.d_model)
        # return self.lut(x) * math.sqrt(self.d_model)


class Sqs2sqs_att(nn.Module):
    def __init__(self, encoder, decoder, input_len=20, output_len=20, embed_dim = 64, dim=2):
        super().__init__()
        self.output_len = output_len
        self.encoder_embed = LinearEmbedding(input_len, embed_dim)
        self.decoder_embed = LinearEmbedding(output_len, embed_dim)
        self.encoder = encoder
        self.decoder = decoder
        mask = torch.tril(torch.ones(output_len, output_len)).unsqueeze(-1).repeat(1, 1, dim)
        self.register_buffer('mask', mask)

    def forward(self, inputs, labels):
        decoder_inputs = self.encoder(self.encoder_embed(inputs.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous())
        # decoder_inputs = output
        labels = labels.roll(1, -2)
        labels[:, 0, :] = inputs[:, -1, :]
        b, s, f = labels.size()
        labels = labels.unsqueeze(-3).repeat((1, s, 1, 1))
        decoder_hidden = labels.masked_fill(self.mask == 0, 0)  # 1e-9
        x = self.decoder(decoder_hidden, decoder_inputs,self.decoder_embed)
        return x

    def evaluate(self, x):
        b, s, f = x.size()
        # labels = torch.zeros_like(x)

        y = x.new_zeros((b,self.output_len,f))
        for i in range(self.output_len):
            data = self.forward(x, y)
            y[:, i, :] = data[:, -1, :]
            # labels = data.unsqueeze(-2)
        return y


def sqs2sqs__att(input_size, output_size,input_len = 20, output_len=20, encoder_hidden=10, decoder_hidden=10, embed_dim=64, encoder_layers=3,
                 heads=1, decoder_layers1 =1, decoder_layers2 =1, dropout=0.):
    encoder = Attention_encoder(output_size=input_size, output_len=embed_dim, hidden_size=encoder_hidden,
                                heads=heads, attention_n=encoder_layers, dropout=0.)
    decoder = Attention_decoder(output_size=output_size, output_len=embed_dim, hidden_size=decoder_hidden,out = output_len,
                                heads=heads, attention_n=decoder_layers1, attention_n1=decoder_layers2, dropout=dropout)
    model = Sqs2sqs_att(input_len =input_len,output_len=output_len,embed_dim = embed_dim, encoder=encoder, decoder=decoder, dim=output_size)
    return model


def sqs2sqs__att_p(input_size, output_size, output_len=20, encoder_features=10, decoder_hidden=10, encoder_layers=3,
                   heads=1, attention_n=1, attention_n1=1, dropout=0.):
    encoder = Attention_encoder_p(output_size=input_size, output_len=output_len, hidden_size=encoder_features,
                                  heads=heads, attention_n=encoder_layers, dropout=0.)
    decoder = Attention_decoder_p(output_size=output_size, output_len=output_len, hidden_size=decoder_hidden,
                                  heads=heads, attention_n=attention_n, attention_n1=attention_n1, dropout=dropout)
    model = Sqs2sqs_att(output_len=output_len, encoder=encoder, decoder=decoder, dim=output_size)
    return model


def sqs2sqs__att_best():
    return sqs2sqs__att(2, 2, 20, 20, 40, 40, 64, 8, 1, 8, 8, 0.1)


def sqs2sqs__att_best_p():
    return sqs2sqs__att_p(2, 2, 20, 40, 40, 8, 1, 8, 8, 0.1)


models = {
    'sqs2sqs__att_best': sqs2sqs__att_best,
    'sqs2sqs__att_best_p': sqs2sqs__att_best_p,
}

if __name__ == '__main__':
    # x = torch.tensor([[[0.3688, 0.3064],
    #                    [1.4474, 0.8739],
    #                    [-1.4729, -1.3729],
    #                    [1.4474, 0.8739]]])  # 1 x 4 x 2
    # sq=sqs2sqs__att(2,2,3,2,2,1,1,1,1,0)
    # sq = sqs2sqs__att_p(2, 2, 4, 2, 2, 1, 1, 1, 1, 0)
    x = torch.randn(3,8,2)
    y = torch.randn(3,12,2)
    TF = sqs2sqs__att(2,2,8,12,32,32,32,8,1,4,4,0.2)
    y1 = TF(x,y)
    y2 = TF.evaluate(x)

    pass
    # x = torch.randn((20,1,2))
    # sq = sqs2sqs__att(2, 2, 20, 20, 10, 10, 3, 1, 3, 2, 0.1)
    # import time
    #
    # # x=x.to('cuda')
    # # sq.to('cuda')
    # start = time.time()
    # with torch.no_grad():
    #     y = sq.evaluate(x)
    # print(time.time() - start)
    # # y = sq.evaluate(x)
    # y1 = sq(x, y)
    # pass
