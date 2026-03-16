gcc -std=c11 -O3 -shared -fPIC anneal.c overlap.c random.c pcg_basic.c -lm -o libanneal.so
