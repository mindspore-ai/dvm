# Device Virtual Machine(DVM)

---

#### 项目介绍
DVM(Device Virtual Machine)是当前业界唯一的微秒级实时AI算子编译和执行框架，可以实现对深度神经网络中的手写或融合算子在运行时根据具体Shape做实时算子编译和执行。通过实时编译技术，使得DVM可以原生支持动态Shape、动态图等动态网络场景的高性能图算融合和算子执行。除此之外，DVM也可以用于静态Shape图算融合、自定义手写算子等其它传统算子执行优化场景。

DVM当前支持Ascend NPU硬件，并覆盖c220、c310系列芯片架构。


#### 环境配置
DVM支持在linux下进行编译执行，并依赖如下环境配置：
+ Ascend NPU硬件： 推荐A2/A3;
+ CANN: 推荐8.3版本。其它就近版本理论可支持，未验证；
+ g++：推荐版本7.3.0。 其它就近版本理论可支持，未验证；
+ python: 推荐3.7以上版本。需要包含numpy包。


#### 编译说明
1. 配置CANN环境变量:
   ```bash
    export ASCEND_CUSTOM_PATH=/<path_to_cann>
    source $ASCEND_CUSTOM_PATH/ascend_toolkit/set_env.sh
   ```

#### 真实上板
2. 配置DVM编译变量:
   ```bash
   cd dvm
   source env.sh
   ```
3. 编译DVM
   ```bash
   make -j32
   ```
4. DVM验证执行。如: ```python examples/01_add.py```


#### 仿真模拟

2. 配置 DVM 编译变量:

   ```bash
   cd dvm
   source env.sh --simulator_name=910B1
   ```

3. 设置仿真环境

   根据所使用的仿真模式，**手动配置对应的动态库路径**：

   * **ESL Model 模式**

     ```bash
     export LD_LIBRARY_PATH=/path/to/your/esl_lib:$LD_LIBRARY_PATH
     ```

   * **常规 Simulator 模式**

     ```bash
     export LD_LIBRARY_PATH=${ASCEND_PATH}/tools/simulator/${DVM_SOC_NAME}/lib:$LD_LIBRARY_PATH
     ```

4. 编译 DVM:

   ```bash
   make -j32
   ```

5. 使用 `msprof` 进行仿真 Profiling，例如：

   ```bash
   msprof op simulator --application="python test_xxx.py" --output=./profiling
   ```


#### 贡献
欢迎参与项目贡献。 具体细节参考MindSpore贡献者Wiki。您可以从多个方面参与项目贡献, 包括并不限于：
+ 提交Bug修复, 避免其他人遇到同样问题；
+ 提交在您具体使用场景遇到的新需求或新问题(issue), 支持我们后续持续演进。不过不能保证最终一定会采纳或按您的设想来实现；
+ 提交您的一些方案或技术想法(issue), 跟我们一起讨论，大家共同进步;
+ 提交新特性支持。 最好先提交issue进行方案讨论，避免拒绝合入或返工，浪费您的宝贵时间;
+ 提交对已有代码的优化改进。作为实时编译器，编译性能是至关重要的指标。如果您对其中一些关键路径代码有更好的优化思路或实现，我们也非常欢迎。


#### 许可证
[Apache License 2.0](https://gitcode.com/mindspore/dvm/blob/master/LICENSE)
