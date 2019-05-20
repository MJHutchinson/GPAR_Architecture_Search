#!/usr/bin/env bash

a=(1 2 3)
b=(A B C)

for index in in ${!a[*]}; do
    echo "${a[$index]} is in ${b[$index]}"
done