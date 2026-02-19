CC = gcc
CFLAGS = -O2 -Wall

# Auto-detect TestU01 location from Homebrew
TESTU01_PREFIX := $(shell brew --prefix testu01 2>/dev/null)

ifeq ($(TESTU01_PREFIX),)
    $(error "TestU01 not found via Homebrew. Run: brew install testu01")
endif

INCLUDE_DIR := $(TESTU01_PREFIX)/include
LIB_DIR     := $(TESTU01_PREFIX)/lib

testu01_bridge: testu01_bridge.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -L$(LIB_DIR) -o testu01_bridge testu01_bridge.c \
		-ltestu01 -lprobdist -lmylib -lm

clean:
	rm -f testu01_bridge