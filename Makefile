VPATH = ./src:./include
OBJ = ops.o kernel.o xkernel.o code.o dvm.o pass.o msprof.o system.o tuning.o comm.o

CFLGAS = --std=c++17 -Werror -Wall -I./include -I./third_party/pybind11/include -I${PY_INCLUDE} -I${ASCEND_PATH}/latest/include -fPIC -fvisibility=hidden
CCE_FLGAS_C220 = --std=c++17 -Wno-int-to-pointer-cast --cce-aicore-only -DAICORE_ARCH_C220 --cce-auto-sync=off -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false  -mllvm -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true -mllvm -cce-aicore-mask-opt=false

ifneq ($(dbg),)
CFLGAS += -g -O0 -DDEBUG
CCE_FLGAS_C220 += -DDEBUG
else
CFLGAS += -O2
endif

ifneq ($(sim),)
LD_FLAGS = -L${ASCEND_PATH}/latest/toolkit/tools/simulator/Ascend$(sim)/lib -lruntime_camodel
CFLGAS += -DVK_SIM_MODEL
CCE_FLGAS_C220 += -g
else
LD_FLAGS = -L${ASCEND_PATH}/latest/lib64 -lascendcl
endif

ifneq ($(asan),)
CFLGAS += -fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer
endif

VMAIN_OFFSET=0x$$(llvm-objdump -t vm_aic_c220.o | grep " vmain_mix_aic$$" | awk '{print $$5}')

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

vm.o: g_vkernel_c220_bin
	echo "extern const" > vm.cc
	xxd -i g_vkernel_c220_bin >> vm.cc
	llvm-objdump -t g_vkernel_c220_bin | grep " F " | python scripts/find_addrs.py src/isa.h >> vm.cc
	g++ -c $(CFLGAS) vm.cc -o vm.o

g_vkernel_c220_bin: vm_aiv_c220.o vm_aic_c220.o
	ld.lld -Ttext=0 vm_aic_c220.o vm_aiv_c220.o -static -o g_vkernel_c220_bin

vm_aiv_c220.o: vm_aiv.cce isa.h vm_aic_c220.o
	ccec -c -O2 $(CCE_FLGAS_C220) -D VMAIN_OFFSET=$(VMAIN_OFFSET) --cce-aicore-arch=dav-c220-vec src/vm_aiv.cce -o vm_aiv_c220.o

vm_aic_c220.o: vm_aic.cce isa.h
	ccec -c -O2 $(CCE_FLGAS_C220) --cce-aicore-arch=dav-c220-cube src/vm_aic.cce -o vm_aic_c220.o

clean:
	rm -f *.o *.so *.a *bin vm.cc

help:
	@echo "Usage: make [sim=910B1|910B2|...] [dbg=1] [asan=1]"
