# Positional encoding (section 5.1)
import torch
import torch.nn as nn

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []  # 用于存储嵌入函数的列表
        d = self.kwargs['input_dims']  # 输入维度
        out_dim = 0  # 嵌入输出的总维度

        # 如果包含输入本身，则将输入作为嵌入的一部分
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)  # 将输入本身添加到嵌入函数中
            out_dim += d  # 增加输出维度

        max_freq = self.kwargs['max_freq_log2']  # 最大频率的对数尺度
        N_freqs = self.kwargs['num_freqs']  # 频率数量
        
        # 根据是否为对数采样来生成频率带
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        # 为每个频率和周期函数（sin, cos）生成嵌入函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d  # 每添加一个函数都增加输出维度
                
        self.embed_fns = embed_fns  # 存储所有的嵌入函数
        self.out_dim = out_dim  # 最终的嵌入输出维度
        
    def embed(self, inputs):
        # 依次执行每个嵌入函数，并将结果沿最后一个维度拼接
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # 如果 i 为 -1，则直接返回恒等映射函数
    if i == -1:
        return nn.Identity(), 3
    
    # 嵌入参数设置，包括是否包含输入本身、输入维度、最大频率、频率数量等
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    # 创建 Embedder 实例
    embedder_obj = Embedder(**embed_kwargs)
    # 定义一个嵌入函数，封装了调用 embedder 对象的方法
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim  # 返回嵌入函数及其输出维度