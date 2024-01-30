VPATH = ./src:./include
OBJ = ops.o kernel.o code.o pybind_api.o dvm.o

CFLGAS = -I./include -I./third_party/pybind11/include -I${PY_INCLUDE} -I${ASCEND_PATH}/latest/include -fPIC -shared
ifneq ($(dbg),)
CFLGAS += -g -O0
else
CFLGAS += -O2
endif

CCE_FLGAS = -Wno-int-to-pointer-cast --cce-aicore-only -DAICORE_ARCH_C100 --cce-aicore-arch=dav-c100
CCE_FLGAS_910B = -Wno-int-to-pointer-cast --cce-aicore-only -DAICORE_ARCH_C220 --cce-aicore-arch=dav-c220-vec --cce-auto-sync=off -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false  -mllvm -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true -mllvm -cce-aicore-mask-opt=false

ifneq ($(sim),)
LD_FLAGS = -L${ASCEND_PATH}/latest/toolkit/tools/simulator/Ascend$(soc)/lib -L${ASCEND_PATH}/latest/lib64 -lruntime_camodel -lascendcl
CFLGAS += -DVK_SIM_MODEL
else
LD_FLAGS = -L${ASCEND_PATH}/latest/lib64 -lruntime -lascendcl
endif

HEADERS = $(OBJ:.o=.h) isa.h acl_ext.h

all: builder.so

builder.so: pybind_api.o libdvm.a
	g++ -shared $^ $(LD_FLAGS) -o $@
	cp $@ ./python/dvm

libdvm.a: ops.o kernel.o code.o dvm.o vm.o
	ar crv $@ $^

${OBJ}: %.o: %.cc $(HEADERS)
	g++ --std=c++17 -Werror -Wall -c $(CFLGAS) $< -o $@

vm.o: vm.cce isa.h
	ccec -c -O2 $(CCE_FLGAS) src/vm.cce -o g_vkernel_bin
	ccec -c -O2 $(CCE_FLGAS_910B) src/vm.cce -o g_vkernel_910b_bin
	xxd -i g_vkernel_bin > vm.cc
	xxd -i g_vkernel_910b_bin >> vm.cc
	g++ --std=c++17 -Werror -Wall -c $(CFLGAS) vm.cc -o vm.o

clean:
	rm *.o *.so *.a

help:
	@echo "Usage: make [sim=1] [dbg=1]"
