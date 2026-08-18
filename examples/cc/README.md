## 简介

本目录提供基于C++接口使用DVM进行算子编译和执行的例子。该算子实现简单乘法和加法融合。其计算逻辑为：
```
fuse_graph(x[1024]:float, y[1024]:float) -> [1024]:float {
  z[1024] = x[1024] * y[1024]
  r[1024] = z[1024] + 0.5
  return r[1024]
}
```

## 编译执行
1. 切换到当前路径下。执行如下编译命令, 完成add程序编译:
   ```bash
   ${CXX:-g++} -I../../include -I${ASCEND_HOME_PATH}/include -L${ASCEND_HOME_PATH}/lib64 -ldl -lascendcl muladd.cc ../../libdvm.a -o add
   ```
2. 执行编译生成的add程序
   ```bash
   ./add
   ```
