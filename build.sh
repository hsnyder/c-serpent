#!/bin/sh
xxd -i preamble > preamble.c
cc -I../useful-c/ -g -Wall -Wextra -Wswitch-enum wrapgen.c -o wrapgen
