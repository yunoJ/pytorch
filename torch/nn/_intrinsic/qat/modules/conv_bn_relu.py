from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
from torch.nn.modules.conv import _ConvNdBase
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch.nn._intrinsic import ConvBn2d as NNConvBn2d
from torch.nn._intrinsic import ConvBnReLU2d as NNConvBnReLU2d
from torch.nn import Parameter
import torch.nn.functional as F


class ConvBn2d(_ConvNdBase):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """
    __FLOAT_MODULE__ = NNConvBn2d

    def __init__(self,
                 # conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 # bn args
                 # num_features: enforce this matches out_channels before fusion
                 eps=1e-05, momentum=0.1,
                 # affine: enforce this is True before fusion?
                 # tracking_running_stats: enforce this is True before fusion
                 # args for this module
                 freeze_bn=False,
                 activation_fake_quant=None,
                 weight_fake_quant=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConvBn2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, False, _pair(0),
                                       groups, bias, padding_mode)
        self.eps = eps
        self.momentum = momentum
        self.freeze_bn = freeze_bn
        self.num_features = out_channels
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.affine = True
        self.track_running_stats = True
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.observer = activation_fake_quant()
        self.weight_fake_quant = weight_fake_quant()
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def enable_fake_quant(self):
        self.observer.enable()
        self.weight_fake_quant.enable()
        return self

    def disable_fake_quant(self):
        self.observer.disable()
        self.weight_fake_quant.disable()
        return self

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        scale_factor = self.gamma / torch.sqrt(self.running_var + self.eps)
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        conv = self.conv2d_forward(input, self.weight_fake_quant(scaled_weight))

        conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
        batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
        batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)

        if not self.freeze_bn:
            conv = conv * (torch.sqrt((self.running_var + self.eps) / (batch_var + self.eps))).reshape([1, -1, 1, 1])
            conv = conv + (self.beta - self.gamma * (batch_mean / torch.sqrt(batch_var + self.eps))).reshape([1, -1, 1, 1])
        else:
            conv = conv + (self.beta - self.gamma * self.running_mean /
                           torch.sqrt(self.running_var + self.eps)).reshape([1, -1, 1, 1])

        self.running_mean = exponential_average_factor * batch_mean + (1 - exponential_average_factor) * self.running_mean
        self.running_var = exponential_average_factor * batch_var + (1 - exponential_average_factor) * self.running_var

        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(ConvBn2d, self).extra_repr()

    def forward(self, input):
        return self.observer(self._forward(input))

    @classmethod
    def from_float(cls, mod, qconfig):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls.__FLOAT_MODULE__, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls.__FLOAT_MODULE__.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig.activation,
                         qconfig.weight)

        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.gamma = bn.weight
        qat_convbn.beta = bn.bias
        qat_convbn.running_mean = bn.running_mean
        qat_convbn.running_var = bn.running_var
        qat_convbn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

class ConvBnReLU2d(ConvBn2d):
        r"""
        A ConvBn2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
        attached with FakeQuantize modules for both output activation and weight,
        used in quantization aware training.

        We combined the interface of :class:`torch.nn.Conv2d` and
        :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

        Implementation details: https://arxiv.org/pdf/1806.08342.pdf

        Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
        default.

        Attributes:
            observer: fake quant module for output activation, it's called observer
                to align with post training flow
            weight_fake_quant: fake quant module for weight

        """
        __FLOAT_MODULE__ = NNConvBnReLU2d

        def __init__(self,
                     # conv2d args
                     in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1,
                     bias=True, padding_mode='zeros',
                     # bn args
                     # num_features: enforce this matches out_channels before fusion
                     eps=1e-05, momentum=0.1,
                     # affine: enforce this is True before fusion?
                     # tracking_running_stats: enforce this is True before fusion
                     # args for this module
                     freeze_bn=False,
                     activation_fake_quant=None,
                     weight_fake_quant=None):
            super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, groups, bias,
                                               padding_mode, eps, momentum,
                                               freeze_bn,
                                               activation_fake_quant,
                                               weight_fake_quant)

        def forward(self, input):
            return self.observer(F.relu(super(ConvBnReLU2d, self)._forward(input)))
