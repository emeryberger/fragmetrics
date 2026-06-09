# Build the standalone fragtrace interposition library.
#
#   make            # build for the host platform
#   make test       # build + run the end-to-end smoke test
#   make clean
#
# Platform is auto-detected: .dylib + DYLD_INTERPOSE on macOS, .so + LD_PRELOAD
# symbol override on Linux. No dependencies beyond a C compiler (and libdl on
# Linux). For the alloc8-based C++ variant, see fragtrace_alloc8.cpp / README.md.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
UNAME   := $(shell uname -s)

ifeq ($(UNAME),Darwin)
  LIB     := libfragtrace.dylib
  BLIB    := libfragtrace_backend.dylib
  LDFLAGS := -dynamiclib
  PRELOAD := DYLD_INSERT_LIBRARIES
else
  LIB     := libfragtrace.so
  BLIB    := libfragtrace_backend.so
  LDFLAGS := -shared
  LIBS    := -ldl
  PRELOAD := LD_PRELOAD
endif

.PHONY: all test clean

all: $(LIB) $(BLIB)

# request-stream tracer (chains to the next malloc in the preload order)
$(LIB): fragtrace.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# backend-selectable tracer (FRAGTRACE_BACKEND=system|mimalloc|jemalloc)
$(BLIB): fragtrace_backend.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

test: $(LIB)
	@echo '#include <stdlib.h>' > /tmp/_fragtest.c
	@echo 'int main(){void*p[8];for(int i=0;i<8;i++)p[i]=malloc(64*(i+1));for(int i=0;i<8;i+=2)free(p[i]);for(int i=1;i<8;i+=2)free(p[i]);return 0;}' >> /tmp/_fragtest.c
	$(CC) -O0 -o /tmp/_fragtest /tmp/_fragtest.c
	FRAGTRACE_OUT=/tmp/_fragtest.jsonl $(PRELOAD)=./$(LIB) /tmp/_fragtest
	@echo "--- captured trace ---"
	@cat /tmp/_fragtest.jsonl
	@python3 -c "import json; [json.loads(l) for l in open('/tmp/_fragtest.jsonl')]; print('OK: valid JSONL')"

clean:
	rm -f $(LIB) /tmp/_fragtest /tmp/_fragtest.c /tmp/_fragtest.jsonl
