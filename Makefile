VPATH = ./src:./include
OBJ = ops.o kernel.o xkernel.o code.o dvm.o pass.o msprof.o system.o tuning.o comm.o dvm_py.o

CFLGAS = --std=c++17 -Werror -Wall -I./include $(PYBIND11_INCLUDES) -I${ASCEND_PATH}/include -fPIC -fvisibility=hidden
CFLGAS += -Wl,-z,relro,-z,now,-z,noexecstack -fstack-protector-all

CCE_FLGAS_C220 = --std=c++17 -Wno-int-to-pointer-cast\
				 --cce-aicore-only\
                 --cce-auto-sync=off \
                 -mllvm -cce-aicore-function-stack-size=16000 \
                 -mllvm -cce-aicore-record-overflow=false \
                 -mllvm -cce-aicore-addr-transform \
                 -mllvm -cce-aicore-jump-expand=true \
                 -mllvm -cce-aicore-mask-opt=false

CCE_FLGAS_C310 = --std=c++17 -Wno-int-to-pointer-cast\
                 --cce-aicore-only \
                 -mllvm -cce-aicore-stack-size=0x8000 \
                 -mllvm -cce-aicore-function-stack-size=0x8000 \
                 -mllvm -cce-aicore-addr-transform \
                 -mllvm -cce-aicore-or-combine=false \
                 -mllvm -instcombine-code-sinking=false \
                 -Xclang -fcce-vf-vl=256 \
                 --cce-auto-sync=off \
                 -mllvm -cce-aicore-jump-expand=true \
                 -mllvm -cce-aicore-mask-opt=false \
                 --cce-simd-vf-fusion=true \
                 -mllvm -cce-aicore-dcci-before-kernel-end=false \
                 -mllvm -cce-aicore-dcci-insert-for-scalar=false


ifneq ($(dbg),)
CFLGAS += -g -O0 -DDEBUG
CCE_FLGAS_C220 += -DDEBUG
CCE_FLGAS_C310 += -DDEBUG
else
CFLGAS += -O2 -D_FORTIFY_SOURCE=2
endif

ifneq ($(DVM_SOC_NAME),)
LD_FLAGS = -L${ASCEND_PATH}/toolkit/tools/simulator/${DVM_SOC_NAME}/lib -lruntime_camodel -L${ASCEND_PATH}/lib64 -lascendcl
CFLGAS += -DVK_SIM_MODEL
else
LD_FLAGS = -L${ASCEND_PATH}/lib64 -lascendcl
endif

ifneq ($(CANN_VER_85),)  # TODO: remove me..
CFLGAS += -D__CANN_85__ -I${ASCEND_PATH}/pkg_inc
C310_ARCH_CUBE=dav-c310-cube
C310_ARCH_VEC=dav-c310-vec
else
C310_ARCH_CUBE=dav-c310
C310_ARCH_VEC=dav-c310
endif

ifneq ($(asan),)
CFLGAS += -fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer
endif

VMAIN_OFFSET=0x$$(llvm-objdump -t vm_aic_c220.o | grep " dvm_mix_aic$$" | awk '{print $$5}')
VMAIN_C310_OFFSET=0x$$(llvm-objdump -t vm_aic_c310.o | grep " dvm_mix_aic$$" | awk '{print $$5}')

HEADERS = $(OBJ:.o=.h) isa.h

all: _dvm_py.so

_dvm_py.so: pybind_api.o dry_run.o libdvm.a
	g++ -shared  $^ $(LD_FLAGS) -o $@
	cp $@ ./python/dvm

libdvm.a: $(OBJ) vm.o
	ar crv $@ $^

pybind_api.o: pybind_api.cc pybind_api.h $(HEADERS)
	g++ -c $(CFLGAS) $< -o $@

dry_run.o: dry_run.cc isa.h vm_aiv.cce vm_aic.cce
	g++ -c $(CFLGAS) $< -o $@

${OBJ}: %.o: %.cc $(HEADERS)
	g++ -c $(CFLGAS) $< -o $@

vm.o: g_vkernel_c220_bin g_vkernel_c310_bin
	echo "extern const" > vm.cc
	xxd -i g_vkernel_c220_bin >> vm.cc
	xxd -i g_vkernel_c310_bin >> vm.cc
	objdump -t g_vkernel_c310_bin | grep " F " | python scripts/find_addrs.py src/isa.h c310 >> vm.cc
	objdump -t g_vkernel_c220_bin | grep " F " | python scripts/find_addrs.py src/isa.h c220 >> vm.cc
	g++ -c $(CFLGAS) vm.cc -o vm.o

ifneq ($(PRE_ASCEND),)
g_vkernel_c220_bin: prebuild/g_vkernel_c220_bin
	cp -f $< $@
g_vkernel_c310_bin: prebuild/g_vkernel_c310_bin
	cp -f $< $@
else
g_vkernel_c220_bin: vm_aiv_c220.o vm_aic_c220.o
	ld.lld -Ttext=0 vm_aic_c220.o vm_aiv_c220.o -static -o g_vkernel_c220_bin
g_vkernel_c310_bin: vm_aiv_c310.o vm_aic_c310.o
	ld.lld -Ttext=0 vm_aic_c310.o vm_aiv_c310.o -static -o g_vkernel_c310_bin
endif

vm_aiv_c220.o: vm_aiv.cce isa.h vm_aic_c220.o
	ccec -c -O2 $(CCE_FLGAS_C220) -D VMAIN_OFFSET=$(VMAIN_OFFSET) --cce-aicore-arch=dav-c220-vec src/vm_aiv.cce -o vm_aiv_c220.o

vm_aic_c220.o: vm_aic.cce isa.h
	ccec -c -O2 $(CCE_FLGAS_C220) --cce-aicore-arch=dav-c220-cube src/vm_aic.cce -o vm_aic_c220.o

vm_aiv_c310.o: vm_aiv_c310.cce isa.h vm_aic_c310.o
	ccec -c -O2 $(CCE_FLGAS_C310) -D VMAIN_OFFSET=$(VMAIN_C310_OFFSET) --cce-aicore-arch=$(C310_ARCH_VEC) src/vm_aiv_c310.cce -o vm_aiv_c310.o

vm_aic_c310.o: vm_aic_c310.cce isa.h
	ccec -c -O2 $(CCE_FLGAS_C310) --cce-aicore-arch=$(C310_ARCH_CUBE) src/vm_aic_c310.cce -o vm_aic_c310.o

clean:
	rm -f *.o *.so *.a *bin vm.cc

help:
	@echo "Usage: make [dbg=1] [asan=1] [PRE_ASCEND=1]"
