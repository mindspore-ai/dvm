# Device Virtual Machine(DVM)


| [用户开发指南](docs/tutorial.md) | [Example示例](examples) | [DVM框架开发](docs/development.md) |

---

#### 项目介绍
DVM(Device Virtual Machine)是当前业界唯一的微秒级实时AI算子编译和执行框架，可以实现对深度神经网络中的手写或融合算子在运行时根据具体Shape做实时算子编译和执行。通过实时编译技术，使得DVM可以原生支持动态Shape、动态图等动态网络场景的高性能图算融合和算子执行。除此之外，DVM也可以用于静态Shape图算融合、自定义手写算子等其它传统算子执行优化场景。当前，DVM已应用于[MindSpore](https://gitcode.com/mindspore/mindspore)、[torch_npu](https://gitcode.com/Ascend/pytorch)等多个下游AI框架，用于实现图模式或Eager模式的自动图算融合优化。对于大部分网络场景，都可获得较为显著的整网融合性能收益。

DVM支持Ascend NPU硬件，并覆盖c220、c310系列芯片架构

#### 环境配置
DVM支持在linux下进行编译执行，并依赖如下环境配置：
+ Ascend NPU硬件： 推荐A2/A3;
+ CANN: 推荐8.3版本。其它就近版本理论可支持，未验证；
+ g++：推荐版本7.3.0。 其它就近版本理论可支持，未验证；
+ python: 推荐3.7以上版本。需要包含numpy、pybind11包。


#### 编译执行
1. 配置CANN环境变量:
   ```bash
    export ASCEND_CUSTOM_PATH=/<path_to_cann>
    source $ASCEND_CUSTOM_PATH/ascend_toolkit/set_env.sh
   ```
2. 配置DVM编译变量:
   ```bash
   cd dvm
   source env.sh
   ```
3. DVM编译, 生成DVM库(libdvm.a)以及pybind接口库(_dvm_py.so)：
   ```bash
   make -j32
   ```
4. DVM验证执行。包括基于DVM相关接口定义算子计算逻辑以及执行算子。 DVM当前支持两种使用方式：
   + 使用python接口: 如 ```python examples/01_add.py```。 这种方式主要用于DVM功能验证或自定义算子表达；
   + 使用C++接口: 需要用户程序包含libdvm.a, 并基于DVM的C++接口进行算子定义和执行。具体示例可参考: [cc example](examples/cc/README.md)


#### 贡献
欢迎参与项目贡献。 具体细节参考MindSpore贡献者Wiki。您可以从多个方面参与项目贡献, 包括并不限于：
+ 提交Bug修复, 帮助其他人遇到同样问题的人避免重复踩坑；
+ 提交在您具体使用场景遇到的新需求或新问题(issue), 支持我们后续持续演进。不过不能保证最终一定会采纳或按您的设想来实现；
+ 提交您的一些方案或技术想法(issue), 跟我们一起讨论，大家共同进步;
+ 提交新特性支持。 最好先提交issue进行方案讨论，避免拒绝合入或返工，浪费您的宝贵时间;
+ 提交对已有代码的优化改进。作为实时编译器，编译性能是至关重要的指标。如果您对其中一些关键路径代码有更好的优化思路或实现，我们也非常欢迎。


#### 许可证
[Apache License 2.0](https://gitcode.com/mindspore/dvm/blob/master/LICENSE)

#### Special Interest Group(SIG)

如果您对DVM相关技术讨论感兴趣，请考虑加入AKG SIG群。

![AKG_SIG](docs/figures/AKG_QRCode.png)
