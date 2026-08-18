# Device Virtual Machine(DVM)


| [技术论文](https://arxiv.org/abs/2603.24239) | [用户手册](docs/tutorial.md) | [如何在TorchNPU中使用](docs/pytorch.md) | [Example示例](examples) | [DVM开发](docs/development.md) |

---

#### 项目介绍
DVM(Device Virtual Machine)是当前业界唯一的微秒级实时AI算子编译和执行框架，可以实现对深度神经网络中的手写或融合算子在运行时根据具体Shape做实时算子编译和执行。通过实时编译技术，使得DVM可以原生支持动态Shape、动态图等动态网络场景的高性能图算融合和算子执行。除此之外，DVM也可以用于静态Shape图算融合、自定义手写算子等其它传统算子执行优化场景。当前，DVM已应用于[MindSpore](https://gitcode.com/mindspore/mindspore)、[TorchNPU](https://gitcode.com/Ascend/pytorch)等多个下游AI框架，用于实现图模式或Eager模式的自动图算融合优化。对于大部分网络场景，都可获得较为显著的整网融合性能收益。

DVM支持Ascend NPU硬件，并覆盖c220、c310系列芯片架构

#### 环境配置
DVM支持在linux下进行编译执行，并依赖如下环境配置：
+ Ascend NPU硬件： A2/A3/A5均可;
+ CANN: 推荐8.5版本及以上；
+ C++编译器：支持GCC和Clang，已验证GCC 7.3.0和Clang 18.1，其它就近版本理论可支持，未验证；
+ python: 推荐3.7以上版本。需要包含numpy、pybind11包。


#### 编译执行
1. CANN环境配置: DVM依赖于CANN，所以编译执行前需先完成CANN环境配置。
   ```bash
    source /<path_to_cann>/ascend_toolkit/set_env.sh
   ```
2. DVM环境配置:
   ```bash
   cd dvm
   source env.sh
   ```
3. DVM编译, 生成DVM库(libdvm.a)以及pybind接口库(_dvm_py.so)：
   ```bash
   make -j32
   ```
   Host侧编译器由`CXX`指定；不设置时使用Make默认的`g++`。例如使用Clang编译：
   ```bash
   CXX=clang++ make -j32
   ```
   集成到其它C++工程时，DVM与上层工程应使用ABI兼容的C++标准库和编译选项。
4. DVM验证执行。包括基于DVM相关接口定义算子计算逻辑以及执行算子。 DVM当前支持两种使用方式：
   + 使用python接口: 如 ```python examples/01_add.py```。 这种方式主要用于DVM功能验证或自定义算子表达；
   + 使用C++接口: 需要用户程序包含libdvm.a, 并基于DVM的C++接口进行算子定义和执行。具体示例可参考: [cc example](examples/cc/README.md)


#### 贡献
欢迎参与项目贡献。 您可以从多个方面参与项目贡献, 包括并不限于：
+ 提交在您具体使用场景遇到的新需求或新问题(issue), 帮助我们尽快识别和解决问题，以扫除您在DVM使用过程中的任何障碍；
+ 提交Bug修复, 帮助其他人遇到同样问题的人避免重复踩坑；
+ 提交新特性支持。 但最好先提交issue进行方案讨论，避免拒绝合入或返工，浪费您的宝贵时间;
+ 提交新方案或技术想法(issue), 参与社区讨论，让DVM未来更好满足不同场景需求;
+ 提交对已有代码的优化改进。作为实时编译器，编译性能是至关重要的指标。如果您对其中一些关键路径代码有更好的优化思路或实现，我们也非常欢迎。

其它未尽事宜，请参考MindSpore贡献者Wiki。


#### 许可证
[Apache License 2.0](https://gitcode.com/mindspore/dvm/blob/master/LICENSE)

#### Special Interest Group(SIG)

如果您对DVM相关技术讨论感兴趣，请考虑加入AKG SIG群。

![AKG_SIG](docs/figures/AKG_QRCode.png)
