# 开发技巧

本文档记录DVM开发者在日常开发、调试等过程中常用技巧和方法。

### 仿真模拟

DVM支持ESL和Simulator两种仿真调试模式，用于对VM Kernel进行软件仿真执行，以便识别性能瓶颈、寻找优化点等。 在完成CANN环境变量配置之后，按照如下步骤进行操作即可：

1. 配置 DVM 编译变量:

   ```bash
   cd dvm
   source env.sh --simulator_name=910B1
   ```

2. 设置仿真环境

   根据所使用的仿真模式，**手动配置对应的动态库路径**：

   * **ESL Model 模式**

     ```bash
     export LD_LIBRARY_PATH=/path/to/your/esl_lib:$LD_LIBRARY_PATH
     ```

   * **常规 Simulator 模式**

     ```bash
     export LD_LIBRARY_PATH=${ASCEND_PATH}/tools/simulator/${DVM_SOC_NAME}/lib:$LD_LIBRARY_PATH
     ```

3. 编译 DVM:

   ```bash
   make -j32
   ```

4. 使用 `msprof` 进行仿真 Profiling，例如：

   ```bash
   msprof op simulator --application="python test_xxx.py" --output=./profiling
   ```
