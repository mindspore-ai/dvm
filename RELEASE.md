# DVM Release Notes

---

### r2.11(2026.9.24: a9b23a6c)
+ add view schedule: slice/concat/split fusion. fractal transpose. dup tiling for unfriendly inner axis.
+ add c310 affinity: add vf fusion support. add simt support(gatherload). c310 insn optimize(viewload/viewstore/broadcast).
+ optimize dynamic shape: add dynamic shape optimize pass support. add lazy tuner support.
+ ops: add custom op support. add int64 support for some op. add permute(spec only).
+ cube: (c220) support ND2NZ for large input stride. Improve MatMul splitk cache budgeting.
+ pass: add dce. add double buffer.
+ kernel: vkernele support lazy codegen to optimize codegen overhead.
+ codegen: remove redundant stuff broadcast. use removepad to optimize broadcastx.
+ tuner: gen cache key with block align. add lazy tuner max table limit.
+ dfx: extract DumpR from Dump. add inspect support.
+ makefile: support no gcc compiler such as clang.
+ refactor: implement padstore by viewstore. support emit view of load/store. add TileRegion and extract TileGen for VectorKernel. add PassOptimizer.

### r2.10(2026.5.14: 76d7c4a1)
+ new kernel support: auto speculation vector kernel. symbolic graph kernel. stagekernel support nested use.
+ view ops: new support of view store. new support of view load x. view load support ub broadcast reuse.
+ c310 enhance: support deterministic. cube adapt. lots of bugfix.
+ performance: support removepad for reduce. optimize scalar pipe of broadcastx.
+ api: remove offset parameter of viewload.
+ refactor: flatten mix kernel stagekernel construct. CodeGenHelper compile optimize. split big file of ops. extract vm_aiv.h/vm_aic.h.
+ support DVM_CUSTOM_FLAGS. CANN 8.5- out of support.

### r2.9(2026.1.28: 3a0dd9c0)
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
+ determinsitic: join optimization
+ core: FlexOp wss inplace, remove simdwidth from tiling, strides_/nd_ reuse between ops, merge SLoad/SStore to Load/Store, support CodeWrap, support vm dryrun, merge StoreStatus to StoreCond

### r2.6(2025.3.4: fba771)
+ vector_mask use count mode
+ determinsitic: based on tile visit
+ ops: add ReduceScatter, AllGatherV2
+ core: remove wrapop

### r2.5(2024.12.9)
+ Add VKernelE
+ ops: add AllReduce, ReduceScatter. MatMul with bias, Trunc, Round, Floor, Ceil
+ support broadcast store

### r2.4(2024.8.13)
+ MatMul: SplitK, unalign, online tuning