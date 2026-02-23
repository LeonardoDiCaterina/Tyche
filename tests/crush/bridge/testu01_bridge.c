/*
 * tests/crush/bridge/testu01_bridge.c
 *
 * Reads uint32 values from stdin and feeds them into TestU01.
 * Compile with:
 *   gcc -O2 -o testu01_bridge testu01_bridge.c -ltestu01 -lprobdist -lmylib -lm
 *
 * Usage:
 *   python ../stream.py --n 100000000 | ./testu01_bridge smallcrush
 *   python ../stream.py --n 100000000 | ./testu01_bridge crush
 *   python ../stream.py --n 100000000 | ./testu01_bridge bigcrush
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unif01.h"
#include "bbattery.h"

static unsigned int next_uint32_from_stdin(void) {
    unsigned int val;
    if (fread(&val, sizeof(unsigned int), 1, stdin) != 1) {
        fprintf(stderr, "Error: stdin exhausted before test completed.\n");
        exit(1);
    }
    return val;
}

static double next_double(void* param, void* state) {
    (void)param; (void)state;
    return (next_uint32_from_stdin() + 0.5) / 4294967296.0;
}

static unsigned long next_bits(void* param, void* state) {
    (void)param; (void)state;
    return (unsigned long)next_uint32_from_stdin();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [smallcrush|crush|bigcrush]\n", argv[0]);
        return 1;
    }

    unif01_Gen* gen = unif01_CreateExternGenBits("Tyche", next_bits);

    if (strcmp(argv[1], "smallcrush") == 0) {
        bbattery_SmallCrush(gen);
    } else if (strcmp(argv[1], "crush") == 0) {
        bbattery_Crush(gen);
    } else if (strcmp(argv[1], "bigcrush") == 0) {
        bbattery_BigCrush(gen);
    } else {
        fprintf(stderr, "Unknown battery: %s\n", argv[1]);
        unif01_DeleteExternGenBits(gen);
        return 1;
    }

    unif01_DeleteExternGenBits(gen);
    return 0;
}