# DVM(Device Virtual Machine)

#### 介绍
**开启算子执行新范式！网络模型加速秘密神器！让友商仰望的黑科技！**


#### 使用说明
1. source env.sh
2. make
3. python test_xxx.py


#### 更新dvm仓代码至mindspore仓
流程：dvm仓代码会预先编译成二进制文件libdvm.a，并通过git lfs上传至mindspore仓。具体步骤：
1. 确保本地已安装git-lfs软件，安装指导可参考[官方文档](https://github.com/git-lfs/git-lfs/wiki/installation)，安装后需执行：
   ```
   git lfs install
   ```

2. 登录[CI构建网站](https://build.mindspore.cn/job/Dvm_Gitee_Version/build?delay=0sec)，`BACKEND_TYPE`在下拉菜单里选择`ASCEND`，然后点击界面下方的Build按钮，手动触发CI构建。CI构建的libdvm.a存放在[此网址](https://repo.mindspore.cn/mindspore/dvm/daily/)，选择最新日期下的构建结果即可。

3. 在mindspore仓下更新dvm的代码。
   ``` shell
   # 1. 更新dvm.h
   cp ~/dvm/include/dvm.h ~/mindspore/mindspore/ccsrc/plugin/device/ascend/kernel/dvm

   # 2. 更新libdvm.a（以2024/02/02的CI构建结果为例）
   ## 更新aarch64版本
   cd ~/mindspore/mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/aarch64
   rm libdvm.a lib_info.txt
   wget https://repo.mindspore.cn/mindspore/dvm/daily/202402/20240202/master_20240202104200_9928302bfbd57493e6d8b6d527c69cf5bb67caa8/ascend/aarch64/libdvm.a
   wget https://repo.mindspore.cn/mindspore/dvm/daily/202402/20240202/master_20240202104200_9928302bfbd57493e6d8b6d527c69cf5bb67caa8/ascend/aarch64/lib_info.txt

   ## 更新x86_64版本
   cd ~/mindspore/mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/x86_64
   rm libdvm.a lib_info.txt
   wget https://repo.mindspore.cn/mindspore/dvm/daily/202402/20240202/master_20240202104200_9928302bfbd57493e6d8b6d527c69cf5bb67caa8/ascend/x86_64/libdvm.a
   wget https://repo.mindspore.cn/mindspore/dvm/daily/202402/20240202/master_20240202104200_9928302bfbd57493e6d8b6d527c69cf5bb67caa8/ascend/x86_64/lib_info.txt

   # 3. 提交代码
   cd ~/mindspore
   git add mindspore/ccsrc/plugin/device/ascend/kernel/dvm/dvm.h
   git add mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/aarch64/libdvm.a -f
   git add mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/aarch64/lib_info.txt
   git add mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/x86_64/libdvm.a -f
   git add mindspore/ccsrc/plugin/device/ascend/kernel/dvm/prebuild/x86_64/lib_info.txt
   ```
