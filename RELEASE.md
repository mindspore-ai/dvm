# DVM Release Notes

---

### r2.9(2026.1.28)
+ vm backend: support c310
+ ops: add ViewLoad, ReduceMax, ReduceMin, AllReduce(Max)
+ cube: default tiling to V2+diagonal_z, support DynMix Broadcast
+ kernel: new support speculative/sequence/Split, parallel support cube and cv. support unify workspace
+ support kernel clone
+ support ScalarRef
+ add dvm_py api header. support jit.kernel
+ api refactor: Unary/Binary/Reduce - op_type to template, unify codegen/launch api, some kerneltype to KernelFlag, remove msprof launch, Config.
+ core: affine broker reshape, VkernelS codegen refactor, merge rootdomain/shapetiling to VectorKernel, System to g_system, implement SliceLoad with ViewLoad, FoldProp for Cube Batchfold
+ add examples, docs, prebuild
+ support CANN 8.5

### r2.7(2025.4.30: 5998ac)
+ ops: add GroupedMatMul, OneHot, AllReduce(bf16)
+ MixKernel: Add DynMixKernel for dynamic shape. Sload/SStore support Broadcast
+ VKernelE: add vector parallel add cv mix fusion
+ determinstic: join optimization
+ core: FlexOp wss inplace, remove simdwidth from tiling, strides_/nd_ reuse between ops, merge SLoad/SStore to Load/Store, support CodeWrap, support vm dryrun, merge StoreStatus to StoreCond

### r2.6(2025.3.4: fba771)
+ vector_mask use count mode
+ determinstic: based on tile visit
+ ops: add ReduceScatter, AllGatherV2
+ core: remove wrapop

### r2.5(2024.12.9)
+ Add VKernelE
+ ops: add AllReduce, ReduceScatter. MatMul with bias, Trunc, Round, Floor, Ceil
+ support broadcast store

### r2.4(2024.8.13)
+ MatMul: SplitK, unalign, online tuning