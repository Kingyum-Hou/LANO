import torch
import torch.nn as nn
import math
import abc
from functools import wraps


def map_elementwise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        container, idx = None, None
        for arg in args:
            if type(arg) in (list, tuple, dict):
                container, idx = type(arg), arg.keys() if type(arg) == dict else len(arg)
                break
        if container is None:
            for value in kwargs.values():
                if type(value) in (list, tuple, dict):
                    container, idx = type(value), value.keys() if type(value) == dict else len(value)
                    break
        if container is None:
            return func(*args, **kwargs)
        elif container in (list, tuple):
            get = lambda element, i: element[i] if type(element) is container else element
            return container(wrapper(*[get(arg, i) for arg in args], 
                                     **{key:get(value, i) for key, value in kwargs.items()}) 
                             for i in range(idx))
        elif container is dict:
            get = lambda element, key: element[key] if type(element) is dict else element
            return {key:wrapper(*[get(arg, key) for arg in args], 
                                **{key_:get(value_, key) for key_, value_ in kwargs.items()}) 
                    for key in idx}
    return wrapper


class Module(torch.nn.Module):
    '''Standard module format.
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cpu')
        elif d == 'gpu':
            self.cuda()
            for module in self.modules():
                if isinstance(module, Module):
                    module.__device = torch.device('cuda')
        else:
            raise ValueError
    
    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float32)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float32
        elif d == 'double':
            self.to(torch.float64)
            for module in self.modules():
                if isinstance(module, Module):
                    module.__dtype = torch.float64
        else:
            raise ValueError

    @property
    def act(self):
        if callable(self.activation):
            return self.activation
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if callable(self.initializer):
            return self.initializer
        elif self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
    @map_elementwise
    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return x
            
class Map(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(Map, self).__init__()
    
    def predict(self, x, returnnp=False):
        x = self._to_tensor(x)
        return self(x).cpu().detach().numpy() if returnnp else self(x)
    
class Algorithm(Module, abc.ABC):
    '''Loss-oriented neural network used as an algorithm based on designing loss.
    '''
    def __init__(self):
        super(Algorithm, self).__init__()
        
    #@final
    def forward(self, x):
        return x
    
    @abc.abstractmethod
    def criterion(self, X, y):
        pass
    
    @abc.abstractmethod
    def predict(self):
        pass


class FNN(Map):
    '''Fully-connected neural network.
    Note that
    len(size) >= 2,
    [..., N1, -N2, ...] denotes a linear layer from dim N1 to N2 without bias,
    [..., N, 0] denotes an identity map (as output linear layer).
    '''
    def __init__(self, size, activation='relu', initializer='default'):
        super(FNN, self).__init__()
        self.size = size
        self.activation = activation
        self.initializer = initializer
        
        self.ms = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, len(self.size) - 1):
            x = self.act(self.ms['LinM{}'.format(i)](x))
        return self.ms['LinM{}'.format(len(self.size) - 1)](x) if self.size[-1] != 0 else x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            if self.size[i] != 0:
                bias = True if self.size[i] > 0 else False
                modules['LinM{}'.format(i)] = nn.Linear(abs(self.size[i - 1]), abs(self.size[i]), bias)
        return modules
    
    def __initialize(self):
        for i in range(1, len(self.size)):
            if self.size[i] != 0: 
                self.weight_init_(self.ms['LinM{}'.format(i)].weight)
                if self.size[i] > 0:
                    nn.init.constant_(self.ms['LinM{}'.format(i)].bias, 0)


class MIONet_periodic(torch.nn.Module):
    '''Multiple-input operator network (periodic).
    '''
    def __init__(self, sizes, activation='relu', initializer='default'):
        super().__init__()
        self.sizes = sizes
        self.sensor = self.sizes[0][0]
        self.activation = activation
        self.initializer = initializer

        self.periodic = []
        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
        
    def forward(self, x, To=10):
        y = [a for a in x]
        b, n = y[0].shape
        output_num = int(y[-1].shape[0]/To)
        for i in self.periodic:
            # 1D
            #y[i] = 2 * math.pi * y[i]
            #y[i] = torch.hstack((torch.cos(y[i]), torch.sin(y[i]), torch.cos(2 * y[i]), torch.sin(2 * y[i])))
            # 2D, reference:"A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data"
            x_ = y[i][..., 0:1]
            y_ = y[i][..., 1:2]
            px_ = 1. # period x
            py_ = 1. # period y
            wx_ = (2. * math.pi / px_ * x_)
            wy_ = (2. * math.pi / py_ * y_)
            y[i] = torch.hstack([torch.cos(wx_), torch.cos(wx_-0.5*math.pi), 
                                 torch.cos(wy_), torch.cos(wy_-0.5*math.pi),
                                 torch.cos(wx_+wy_), torch.cos(wx_+wy_-0.5*math.pi),
                                 torch.cos(wx_-wy_), torch.cos(wx_-wy_-0.5*math.pi),
                                 ])
        branch_ = torch.stack([self.ms['Net{}'.format(i + 1)](y[i]) for i in range(len(self.sizes)-2                 )])
        trunk_  = torch.stack([self.ms['Net{}'.format(i + 1)](y[i]) for i in range(len(self.sizes)-2, len(self.sizes))])
        
        output = torch.einsum('Bbd, Tnd -> bn', branch_, trunk_)
        output = output.reshape(b, output_num, To) + self.ps['bias']
        return output
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            size = self.sizes[i]
            if size[0] == 'p':
                # 1D
                #size = [4] + size[1:]
                # 2D
                size = [8] + size[1:]
                self.periodic.append(i)
            modules['Net{}'.format(i + 1)] = FNN(size, self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters
