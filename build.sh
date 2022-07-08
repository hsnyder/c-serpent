#!/bin/sh
xxd -i preamble > preamble.c
xxd -i example > example.c
cc -g -Wall -Wextra -Wswitch-enum wrapgenx.c -o wrapgenx
