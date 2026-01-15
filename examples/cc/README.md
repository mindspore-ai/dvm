## 简介

本目录提供基于C++接口使用DVM进行算子编译和执行的例子。该算子实现简单加法操作。

## 编译执行
1. 切换到当前路径下。执行如下编译命令, 完成add程序编译:
   ```bash
   g++ -I../../include -I${ASCEND_PATH}/include -L${ASCNED_PATH}/lib64 -ldl -lascendcl add.cc ../../libdvm.a -o add
   ```
2. 执行编译生成的add程序
   ```bash
   ./add
   ```
