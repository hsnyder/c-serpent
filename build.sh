#!/bin/sh
xxd -i preamble > preamble.c
xxd -i example > example.c
cc -I../useful-c/ -g -Wall -Wextra -Wswitch-enum wrapgenx.c -o wrapgenx
