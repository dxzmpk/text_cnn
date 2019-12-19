import torch.nn as nn
import preprocessing
import torch.nn.functional as F
import torch




class Text_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        kernel_list = [2,3]
        kernel_num = 3
        self.embed, num_embeddings, embedding_dim = preprocessing.create_emb_layer(True)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (kernel_size, embedding_dim)) for kernel_size in kernel_list])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_list)*kernel_num, args['classes'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.tensor(x).to(torch.int64)
        x = self.embed(x) # 样本数，句子最大长度，词向量维度
        x = x.unsqueeze(1) # 样本数，通道数，句子最大长度，词向量维度
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        y_pred = self.fc1(x)  # (N, C)
        # y_pred = max(list(y_pred))
        # y_pred = torch.LongTensor(y_pred.index(y_pred))
        return y_pred
        

# args = {'classes':2}
# text = torch.LongTensor([[1,2,2,3], [2,3,4,6], [3,4,5,6]])
# net = Text_CNN(args)
# out = net(text)
# print(out)